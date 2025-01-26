from typing import *
import torch
import transformers


class BeamSearch:

    def __init__(self, max_seq_len: int, vocab_size: int, beam_width: int = 2):
        super(BeamSearch, self).__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.beam_width = beam_width

    def search(self, model, images):
        # images: (batch_size, C, W, H)
        # model.eval() should be called outside this scope.

        batch_size = images.size(0)
        
        beam_scorer = transformers.generation.BeamSearchScorer(batch_size, self.beam_width, images.device, 
                                                               max_length=self.max_seq_len)

        beam_scores = torch.zeros((batch_size, self.beam_width), dtype=torch.float, device=images.device)
        beam_scores = beam_scores.view(-1)  # (batch_size * beam_width, )

        input_ids = torch.full((batch_size * self.beam_width, 1), 3, dtype=torch.long, device=images.device)
            
        contexts: Dict[str, torch.Tensor] = model.predict_init(images)
        for k, v in contexts.items():
            contexts[k] = v.repeat_interleave(self.beam_width, dim=0)

        cur_len = 1
        while cur_len < self.max_seq_len - 1:

            outputs, scores, next_contexts = model.predict_next(input_ids[:, -1], contexts)

            scores = torch.nn.functional.log_softmax(scores, dim=-1)  # (batch_size * beam_width, vocab_size)

            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(batch_size, self.beam_width * self.vocab_size)

            next_scores, next_tokens = torch.topk(next_scores, 2 * self.beam_width, dim=1, largest=True, sorted=True)
            next_tokens = next_tokens % self.vocab_size
            next_indices = torch.div(next_tokens, self.vocab_size, rounding_mode="floor")

            # print(next_scores, next_tokens, next_indices)

            r = beam_scorer.process(input_ids, next_scores, next_tokens, next_indices, 
                                    pad_token_id=0, eos_token_id=4)
            # print(r)

            beam_scores = r["next_beam_scores"]
            beam_next_tokens = r["next_beam_tokens"]
            beam_idx = r["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            for k, v in next_contexts.items():
                contexts[k] = v[beam_idx, :]

            if beam_scorer.is_done:
                break

            cur_len += 1

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=0,
            eos_token_id=4,
            max_length=self.max_seq_len
        )

        return sequence_outputs["sequences"]