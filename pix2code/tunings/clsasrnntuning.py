from typing import *

import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


class InjectModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_inputs(self, *inputs):
        return inputs
    
    def handle_outputs(self, outputs):
        return outputs
    

class HijackModule(nn.Module):

    def __init__(self, 
                 hijack_model: nn.Module, 
                 fn_handle_inputs: Optional[Callable] = None, 
                 fn_handle_output: Optional[Callable] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hijack_model = hijack_model
        self.fn_handle_inputs = fn_handle_inputs
        self.fn_handle_output = fn_handle_output

    def forward(self, *inputs):
        if self.fn_handle_inputs is not None:
            inputs = self.fn_handle_inputs(*inputs)
        outputs = self.hijack_model(*inputs)
        if self.fn_handle_output is not None:
            outputs = self.fn_handle_output(outputs)
        return outputs
    

class RnnClsHead(InjectModule):

    def __init__(self, 
                 cls_head: nn.Module, 
                 in_features: int,
                 out_features: int,
                 dropout: float = 0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_head = cls_head
        self.in_features = in_features
        self.out_features = out_features
        self.reset()

        self.embedding = nn.Embedding(90, self.in_features)  # embedding layer
        self.norm = nn.LayerNorm(self.in_features)
        self.rnn_step = nn.LSTMCell(self.in_features * 2, self.in_features)
        self.init_h = nn.Linear(self.in_features, self.in_features)
        self.init_c = nn.Linear(self.in_features, self.in_features)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.in_features, self.out_features)

        self.init_weights()

    def init_weights(self):
        # 
        for name, param in self.rnn_step.named_parameters():
            if "weight_ih" == name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" == name:
                nn.init.orthogonal_(param.data)
            elif "bias_ih" == name:
                param.data.fill_(0)
                # set forget-gate bias to 1
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1)
            elif "bias_hh" == name:
                param.data.fill_(0)
            else:
                assert False, name

        #
        # nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
        #
        for each in [self.fc, self.init_h, self.init_c]:
            nn.init.xavier_uniform_(each.weight.data)
            each.bias.data.fill_(0)

    def init_hidden_state(self, x):
        h = self.init_h(x)
        c = self.init_c(x)
        return h, c

    def reset(self):
        self.buffer_list = []
        self.buffer = None
        self.scores = None

    def forward(self, x): 
        if len(self.buffer_list) == 0:
            self.buffer_list.append(x)
        else:
            b = torch.zeros_like(self.buffer_list[-1]).to(x.device)
            b[:x.size(0)] = x
            self.buffer_list.append(b)

        y = self.cls_head(x)    
        # print(x.shape, y.shape)
        return y

    def self_forward(self, batch):
        if self.training:
            return self.self_forward_train(batch)
        return self.self_forward_eval(batch)

    def self_forward_eval(self, batch):
        batch_size = self.scores.size(0)
        max_decode_length = self.scores.size(1)
        # print("buffer:", self.buffer.shape)
        # print("scores:", self.scores.shape)

        # buffer: torch.Tensor = rearrange(self.buffer, "b s f -> (b s) f")
        # scores: torch.Tensor = rearrange(self.scores, "b s v -> (b s) v")

        target = batch["code"][:, 1:max_decode_length + 1]
        target_len = batch["code_len"] - 1
        print(target)
        # print(target_len)

        # target: torch.Tensor = rearrange(target, "b s l -> (b s) l")
        # target_len: torch.Tensor = rearrange(target_len, "b s -> (b s)")

        # target_len_sorted, sort_ind = target_len.sort(dim=0, descending=True)
        # target_len_sorted = (target_len_sorted - 1).tolist()
        # print(target_len_sorted)
        # print(batch["code_train"]) 

        # buffer_sorted = self.norm(buffer[sort_ind])
        # scores_sorted = scores[sort_ind]
        # target_sorted = target[sort_ind]

        inputs = torch.full((1, ), 3).to(target.device)
        emb = self.embedding(inputs)
        # print(emb.shape)
        emb = repeat(emb, "b d -> (s b) d", s=target.size(1))
        # print(emb.shape)

        pred_scores = torch.zeros((target.size(0), target.size(1), self.scores.size(-1)),
                                  dtype=torch.float).to(self.scores.device)

        for t in range(batch_size):
            buffer = self.buffer[t, :target_len[t], :]
            h, c = self.init_hidden_state(buffer)
            h, c = self.rnn_step(torch.cat([emb[:target_len[t], :], buffer], dim=-1), (h, c))
            
            preds = self.fc(self.dropout(h))
            pred_scores[t, :target_len[t], :] = preds  # + self.scores[t, :target_len[t], :]

        return (
            rearrange(pred_scores, "b s v -> (b s) v"), 
            rearrange(target, "b s -> (b s)"))   
         
    def self_forward_train(self, batch):
        max_decode_length = self.scores.size(1)
        # print("buffer:", self.buffer.shape)
        # print("scores:", self.scores.shape)

        buffer: torch.Tensor = rearrange(self.buffer, "b s f -> (b s) f")
        scores: torch.Tensor = rearrange(self.scores, "b s v -> (b s) v")

        target = batch["code_train"][:, 1:max_decode_length + 1, :]
        target_len = batch["code_lt_len"][:, 1:max_decode_length + 1]
        # print(target)
        # print(target_len)

        
        target: torch.Tensor = rearrange(target, "b s l -> (b s) l")
        target_len: torch.Tensor = rearrange(target_len, "b s -> (b s)")

        inputs = torch.full((target.size(0), 1), 3).to(target.device)
        inputs = torch.cat([inputs, target[:, :-1]], dim=-1)
        # print(inputs)

        target_len_sorted, sort_ind = target_len.sort(dim=0, descending=True)
        target_len_sorted = target_len_sorted.tolist()
        # print(target_len_sorted)
        # print(batch["code_train"]) 

        buffer_sorted = self.norm(buffer[sort_ind])
        scores_sorted = scores[sort_ind]
        target_sorted = target[sort_ind]
        inputs_sorted = inputs[sort_ind]

        pred_scores = torch.zeros((target.size(0), target.size(-1), scores.size(-1)),
                                  dtype=torch.float).to(scores.device)

        h, c = self.init_hidden_state(buffer_sorted)

        for t in range(max(target_len_sorted)):
            batch_size_t = sum([l > t for l in target_len_sorted])

            emb = self.embedding(inputs_sorted[:batch_size_t, t])

            h, c = self.rnn_step(torch.cat([emb, buffer_sorted[:batch_size_t, :]], dim=1), 
                                 (h[:batch_size_t, :], c[:batch_size_t, :]))
            
            preds = self.fc(self.dropout(h))
            pred_scores[:batch_size_t, t, :] = preds  # + scores_sorted[:batch_size_t, :]

        return (
            rearrange(pred_scores, "b l v -> (b l) v"), 
            rearrange(target_sorted, "b l -> (b l)"))        


class ClsAsRnnTuning(nn.Module):

    def __init__(self, 
                 model: nn.Module, 
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.model = model
        self.freeze()
        self.rnn_cls_head = self.inject()

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Tunable parameters: {params:,}")

    @staticmethod
    def rebuild(this: RnnClsHead, outputs):        
        this.buffer = torch.stack(this.buffer_list, dim=1)
        this.buffer = this.buffer[outputs[-1]]
        this.buffer_list = []
        this.scores = outputs[0][outputs[-1]]
        return outputs

    def inject(self):
        model_cls_name = self.model.__class__.__name__
        if model_cls_name == "ImageCaption":
            old_cls_head: nn.Linear = self.model.decoder.fc
            rnn_cls_head = RnnClsHead(old_cls_head, 
                                      old_cls_head.in_features, 
                                      old_cls_head.out_features)
            self.model.decoder.fc = rnn_cls_head
            fn_handle_output = partial(ClsAsRnnTuning.rebuild, rnn_cls_head)
            self.model.decoder = HijackModule(self.model.decoder, 
                                              fn_handle_output=fn_handle_output)
        else:
            assert False, model_cls_name
        return rnn_cls_head

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, batch):
        batch["code"] = batch["code"].long()
        if self.training:
            batch["code_train"] = batch["code_train"].long()
            batch_model = {
                "image": batch["image"],
                "code": batch["code"],
                "code_len": batch["code_len"]
            }
        else:
            batch_model = batch

        self.model(batch_model)

        scores, target = self.rnn_cls_head.self_forward(batch)
        scores = scores[target != 0]
        target = target[target != 0]
        loss = self.criterion(scores, target) 

        self.rnn_cls_head.reset()

        return {
            "loss": loss,
            "logits": scores,
            "scores": nn.functional.softmax(scores, dim=-1),
            "targets": target
        }