from typing import *
import torch


class GreedySearch:

    def __init__(self, max_seq_len: int):
        super(GreedySearch, self).__init__()
        self.max_seq_len = max_seq_len

    def search(self, model, images):
        # images: (batch_size, C, W, H)
        # model.eval() should be called outside this scope.

        batch_size = images.size(0)

        inputs = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long).to(images.device)
        inputs[:, 0] = 3  # <start>

        mask = torch.ones((batch_size, ), dtype=torch.bool)

        contexts: Dict[str, torch.Tensor] = model.predict_init(images)

        for t in range(1, self.max_seq_len - 1):
            if mask.sum() == 0:
                break

            selected_inputs = inputs[mask, t - 1]
            selected_contexts = { k: v[mask] for k, v in contexts.items() }

            outputs, scores, next_contexts = model.predict_next(selected_inputs, selected_contexts)

            inputs[mask, t] = outputs
            for k, v in next_contexts.items():
                contexts[k][mask] = v

            indices = torch.where(torch.logical_or(inputs[:, t] == 0, inputs[:, t] == 4))  # <pad> or <end>
            mask[indices] = 0

        mask[:] = 0
        indices = torch.where(torch.logical_and(inputs[:, -2] != 0, inputs[:, -2] != 4))  # <pad> or <end>
        mask[indices] = 1
        inputs[mask, -1] = 4  # <end>        

        return inputs
            