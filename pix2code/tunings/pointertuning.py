from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import PointerNet


def masked_accuracy(output, target, mask):
	"""Computes a batch accuracy with a mask (for padded sequences) """
	with torch.no_grad():
		masked_output = torch.masked_select(output, mask)
		masked_target = torch.masked_select(target, mask)
		accuracy = masked_output.eq(masked_target).float().mean()
		return accuracy
     

class PointerTuning(nn.Module):

    def __init__(self, 
                 model: nn.Module, 
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.model = model
        self.freeze()
        self.pn = PointerNet(90, 512, 512)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 
        # model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print(f"Tunable parameters: {params:,}")

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, batch):
        outputs = self.model(batch)

        log_pointer_score, argmax_pointer, mask = self.pn(batch["pre_ivs"], batch["pre_les"])
        unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
        loss_pn_pre = F.nll_loss(unrolled, batch["pre_ids"].long().view(-1), ignore_index=-1)

        mask = mask[:, 0, :]
        acc_pre = masked_accuracy(argmax_pointer, batch["pre_ids"], mask).item(), mask.int().sum().item()

        log_pointer_score, argmax_pointer, mask = self.pn(batch["pos_ivs"], batch["pre_les"])
        unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
        loss_pn_pos = F.nll_loss(unrolled, batch["pos_ids"].long().view(-1), ignore_index=-1)

        mask = mask[:, 0, :]
        acc_pos = masked_accuracy(argmax_pointer, batch["pos_ids"], mask).item(), mask.int().sum().item()

        outputs["loss"] += loss_pn_pre + loss_pn_pos
        outputs["loss/pre/pn"] = loss_pn_pre
        outputs["loss/pos/pn"] = loss_pn_pos
        outputs["loss/pre/acc"] = acc_pre[0]
        outputs["loss/pos/acc"] = acc_pos[0]

        return outputs