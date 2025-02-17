from typing import *
import torch
import transformers


class BeamSearch:

    def __init__(self, max_seq_len: int, vocab_size: int, do_sample: bool = False, beam_width: int = 2):
        super(BeamSearch, self).__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.do_sample = do_sample
        self.beam_width = beam_width

    def search(self, model, images, init_input = None):
        # images: (batch_size, C, W, H)
        # model.eval() should be called outside this scope.

        batch_size = images.size(0)
        
        beam_scorer = transformers.generation.BeamSearchScorer(
            batch_size=batch_size, 
            num_beams=self.beam_width, 
            device=images.device, 
            max_length=self.max_seq_len
        )

        # raw_outputs = ()
        raw_logits = ()
        beam_indices = tuple(() for _ in range(batch_size * self.beam_width))

        beam_scores = torch.zeros((batch_size, self.beam_width), dtype=torch.float, device=images.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # (batch_size * beam_width, )

        if init_input is None:
            input_ids = torch.full((batch_size * self.beam_width, 1), 3, dtype=torch.long, device=images.device)
        else:
            input_ids = init_input[:, None].repeat_interleave(self.beam_width, dim=0)
        # print("input_ids", input_ids.shape)

        contexts: Dict[str, torch.Tensor] = model.predict_init(images)
        for k, v in contexts.items():
            contexts[k] = v.repeat_interleave(self.beam_width, dim=0)

        cur_len = 1
        while cur_len < self.max_seq_len - 1:

            outputs, scores, next_contexts = model.predict_next(input_ids, contexts)

            next_token_logits = scores.clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)  
            # (batch_size * beam_width, vocab_size)
            # print(scores.shape)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # raw_outputs += (outputs, )
            raw_logits += (next_token_logits, )

            next_token_scores = next_token_scores.view(batch_size, self.beam_width * self.vocab_size)

            n_tokens_to_keep = 2 * self.beam_width
            if self.do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True)

            next_indices = torch.div(next_tokens, self.vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % self.vocab_size

            # print(next_token_scores, next_tokens, next_indices)

            beam_outputs = beam_scorer.process(
                input_ids, 
                next_token_scores, 
                next_tokens, 
                next_indices, 
                pad_token_id=0, 
                eos_token_id=4,
                beam_indices=beam_indices
            )
            # print(r)

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            
            for k, v in next_contexts.items():
                contexts[k] = v[beam_idx, :]

            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            cur_len += 1

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=0,
            eos_token_id=4,
            max_length=self.max_seq_len,
            beam_indices=beam_indices,
        )

        # print(sequence_outputs)

        # print([torch.argmax(each, dim=-1) for each in raw_logits])
        # print(raw_outputs)

        sequences = sequence_outputs["sequences"]

        length = torch.zeros((batch_size, ), dtype=torch.int)
        output_scores = torch.zeros((batch_size, self.max_seq_len, self.vocab_size), dtype=torch.float).to(images.device)
        for bi in range(batch_size):
            for ti in range(sequences.size(1)):
                # output_scores[bi, ti, :] = -16.118
                output_scores[bi, ti, sequences[bi, ti]] = 1  # -8.9e-6

                if sequences[bi, ti] != 0:
                    length[bi] += 1

        return sequences, output_scores, length