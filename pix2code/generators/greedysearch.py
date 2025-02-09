from typing import *
import torch


class GreedySearch:

    def __init__(self, max_seq_len: int, vocab_size: int, conditions = None):
        super(GreedySearch, self).__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.conditions = conditions

    def search(self, model, images, init_input = None):
        # images: (batch_size, C, W, H)
        # model.eval() should be called outside this scope.

        if self.conditions is not None:
            emb = torch.nn.Embedding(90, 90, _weight=torch.from_numpy(self.conditions), _freeze=True)
            emb = emb.to(images.device)

        batch_size = images.size(0)

        inputs = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long).to(images.device)
        if init_input is None:
            inputs[:, 0] = 3  # <start>
        else:
            inputs[:, 0] = init_input

        output_scores = torch.zeros((batch_size, self.max_seq_len - 1, self.vocab_size), dtype=torch.float).to(images.device)

        mask = torch.ones((batch_size, ), dtype=torch.bool)

        contexts: Dict[str, torch.Tensor] = model.predict_init(images)

        for t in range(1, self.max_seq_len - 1):
            if mask.sum() == 0:
                break

            selected_inputs = inputs[mask, :t]
            selected_contexts = { k: v[mask] for k, v in contexts.items() }

            outputs, scores, next_contexts = model.predict_next(selected_inputs, selected_contexts)

            if self.conditions is not None:
                c = emb(selected_inputs)
                # print(c.shape)
                # import json
                # with open("/Volumes/Home/codehub/pytorch_pix2code/unit_test/vocabs.json", "r") as input:
                #     vocabs = json.load(input)
                # for each_c, each_i, each_o in zip(c, selected_inputs, outputs):
                #     idx = torch.where(each_c.squeeze(0) != 0)[0]
                #     print(vocabs[each_i], "=>", vocabs[each_o], [vocabs[each] for each in idx])
                scores = scores * c
                outputs = torch.argmax(scores, dim=-1)

            inputs[mask, t] = outputs
            output_scores[mask, t - 1] = scores
            for k, v in next_contexts.items():
                contexts[k][mask] = v

            indices = torch.where(torch.logical_or(inputs[:, t] == 0, inputs[:, t] == 4))  # <pad> or <end>
            mask[indices] = 0

        mask[:] = 0
        indices = torch.where(torch.logical_and(inputs[:, -2] != 0, inputs[:, -2] != 4))  # <pad> or <end>
        mask[indices] = 1
        inputs[mask, -1] = 4  # <end>   
        output_scores[mask, -1, 4] = 1.     

        return inputs, output_scores
            