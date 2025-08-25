from typing import *

import numpy as np
import math
import copy
from weakref import proxy
from apted import APTED, Config

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .mlp import MLP
from ..tree import TreeNode
from .ui2box import (
    build_backbone, TransformerEncoderLayer, TransformerEncoder,
    TransformerDecoderLayer, TransformerDecoder, TokenEmbedding, PositionalEncoding
)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


#


def _mask_pads(target, ll, smooth_obj):
    pad_mask = target.eq(0)
    if pad_mask.any():
        ll.masked_fill_(pad_mask, 0.0)
        smooth_obj.masked_fill_(pad_mask, 0.0)
    return ll.squeeze(-1), smooth_obj.squeeze(-1)


class CustomConfig(Config):
   
    def rename(self, node1, node2):
        """Compares attribute .value of trees"""
        return 1 if node1.iv != node2.iv else 0

    def children(self, node):
        """Get left and right children of binary tree"""
        return node.children
    

def compute_ted(src_tree, tgt_tree):
    apted = APTED(src_tree, tgt_tree, CustomConfig())
    ted = apted.compute_edit_distance()
    mapping = apted.compute_edit_mapping() 
    return ted, mapping


def _build_list(n: "TreeNode", fn, ivs: List[int]):
    ivs.append(fn(n, n.iv))
    if len(n.children) > 0:
        ivs.append(fn(None, 5))  # [LB]
        for each in n.children:
            if each.get_iv_premitive() in [3, 4]:
                continue
            _build_list(each, fn, ivs)
        ivs.append(fn(None, 6))  # [RB]
            

def build_list(n, fn) -> List[int]:
    ivs = [fn(None, 3)]  # [START]
    for each in n.children:
        if each.get_iv_premitive() in [3, 4]:
            continue
        _build_list(each, fn, ivs)
    ivs.append(fn(None, 4))  # [END]
    return ivs


def print_list(n: str, list: List[Any], pad: int = 5, ignore_idx = None):
    padding: str = "{each:>" + str(pad) + "}"
    if ignore_idx is None:
        print(f"{n}: {' '.join([padding.format(each=each) for each in list])}")
    else:
        print(f"{n}: {' '.join([padding.format(each=each) for each in list if each != ignore_idx])}")


def make_default(n, v):
    return v


def make_default_label(n, v):
    return v if n else -1


def make_delete_label(n, v):
    if n:
        return 0 if n.align else 1
    return -1


def make_update_label(n, v):
    return n.align.iv if n else -1


MAX_SEQ_LEN = 450


class Encoder(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed):
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, pos=pos_embed)
        return memory, pos_embed
    

class Decoder(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, tgt, tgt_mask, query_embed, pos_embed):
        # print("tgt:", tgt.shape)
        # print("query_embed:", query_embed.shape)
        # print("tgt_mask:", tgt_mask.shape)
        # flatten NxCxHxW to HWxNxC
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = tgt.permute(1, 0, 2)
        # tgt_mask = tgt_mask.permute(1, 0)
        query_embed = query_embed.permute(1, 0, 2)

        hs = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)
    

class ResFormer(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.backbone = build_backbone("resnet50")
        self.encoder = Encoder()
        self.decoder = Decoder()
        hidden_dim = self.encoder.d_model
        self.token_embed = TokenEmbedding(vocab_size, hidden_dim)
        self.query_embed = PositionalEncoding(hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, image, code):
        memory, pos = self.encode(image)
        return self.decode(memory, code, pos)
    
    def encode(self, image):
        features, pos = self.backbone(image)
        src = features[-1]
        inp = self.input_proj(src)
        memory, pos_embed = self.encoder(inp, pos[-1])
        return memory, pos_embed
    
    def decode(self, memory, pos, code):
        tgt = self.token_embed(code)
        query = self.query_embed(tgt)
        hs = self.decoder(memory, tgt, code == 0, query, pos)
        return hs[0]
    

class TreeEditNet2(nn.Module):

    def __init__(self, 
                 vocab_size,
                 max_len,

                 backbone=None, 
                 generator=None,
                 
                 dim=256, 

                 mask_rate = None,
                 min_insert = None,
                 half_train = None,
                 mask_fill = None,
                 x2 = None,
                 mlp = None,
                 balanced = None,

                 verbose = None,
                 proof_of_concept: bool = False):
        
        super().__init__()
        self.proof_of_concept = proof_of_concept
        self.verbose = (verbose == "1")

        self.mask_rate = (float(mask_rate) if mask_rate is not None else 0.0)
        self.min_insert = (int(min_insert) if min_insert is not None else 0)
        self.half_train = (half_train == "1")
        self.mask_fill = (mask_fill == "1")
        self.x2 = (x2 == "1")
        self.mlp = (int(mlp) if mlp is not None else None)
        self.balanced = (balanced == "1")

        print({
            "mask_rate": self.mask_rate,
            "min_insert": self.min_insert,
            "half_train": self.half_train,
            "mask_fill": self.mask_fill,
            "x2": self.x2,
            "balanced": self.balanced,
            "verbose": self.verbose
        })

        self.backbone = backbone
        if backbone is not None:
            self.backbone.eval()
            self.generator = generator(max_seq_len=max_len)
        
        self.bottleneck = ResFormer(vocab_size)
        if self.x2:
            self.bottleneck_x2 = ResFormer(vocab_size)

        if self.mlp is None:
            self.delete_head = nn.Linear(dim, 2)
            # self.insert_head = nn.Linear(dim, 3)
            self.update_head = nn.Linear(dim, vocab_size)
        else:
            self.delete_head = MLP(dim, 256, 2, self.mlp)
            self.update_head = MLP(dim, 256, vocab_size, self.mlp)

        self.criterion_delete = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_update = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch):

        #

        images = batch["image"]
        targets = batch["code"].long()  # (B, S)
        # targets_lens = batch["code_len"]
        batch_size = targets.size(0)
        half_train = getattr(self, "half_train", False)

        #

        # self.backbone.to(images.device)
        if self.backbone is not None:
            with torch.no_grad():
                predicts, _, _ = self.generator.search(self.backbone, batch)
                predicts = predicts.detach().cpu()
        else:
            predicts = batch["pred"]

        #

        if not half_train:
            sources_delete = torch.zeros((batch_size, MAX_SEQ_LEN), dtype=torch.long)
            targets_delete = torch.full_like(sources_delete, -1)

            sources_update = torch.zeros_like(sources_delete)
            targets_update = torch.full_like(sources_delete, -1)

        sources_insdel_list = []
        targets_insdel_list = []
        sources_insupd_list = []
        targets_insupd_list = []

        for i, (src, dst) in enumerate(zip(predicts, targets)):

            if self.proof_of_concept and self.verbose:
                print("-" * 80)
                print_list("--SOURCE", src, ignore_idx=0)
                print_list("--TARGET", dst, ignore_idx=0)

            tree_src = TreeNode.build_tree(src)
            tree_dst = TreeNode.build_tree(dst)

            if not half_train:
                _, mapping = compute_ted(tree_src, tree_dst)

                # 

                for node_src, node_dst in mapping:
                    if node_dst is None:  # DELETE
                        node_src.align = None
                    elif node_src is None:  # INSERT
                        node_dst.align = None
                    else:  # UPDATE
                        node_src.align = proxy(node_dst)
                        node_dst.align = proxy(node_src)

                source_delete = build_list(tree_src, make_default)
                target_delete = build_list(tree_src, make_delete_label)
                # if len(target_delete) > 307:
                #     print_list(f"{len(src)}", tree_src.build_list())
                #     print_list(f"{len(dst)}", tree_dst.build_list())

                if self.training and getattr(self, "balanced", False):
                    if self.proof_of_concept and self.verbose:
                        print_list("S-DELETE", sources_delete[i], ignore_idx=0)
                        print_list("T-DELETE", target_delete)
                        print(target_delete.count(0), target_delete.count(1))
                        
                    td = target_delete
                    td_cnt_0, td_cnt_1 = td.count(0), td.count(1)
                    if td_cnt_0 > td_cnt_1:
                        td = np.array(td)
                        pp = np.random.choice(np.where(td == 0)[0], td_cnt_0 - td_cnt_1) 
                        # print(pp)
                        td[pp] = -1
                        # print(td)
                        target_delete = td.tolist()
                        # assert False

                sources_delete[i, :len(source_delete)] = torch.LongTensor(source_delete)
                targets_delete[i, :len(target_delete)] = torch.LongTensor(target_delete)

                if self.proof_of_concept and self.verbose:
                    print_list("S-DELETE", sources_delete[i], ignore_idx=0)
                    print_list("T-DELETE", target_delete)
                    print(target_delete.count(0), target_delete.count(1))
                    # assert False

                #

                for node_src, node_dst in mapping:
                    if node_dst is None:  # DELETE
                        node_src.delete()

                source_update = build_list(tree_src, make_default)
                target_update = build_list(tree_src, make_update_label)
                
                sources_update[i, :len(source_update)] = torch.LongTensor(source_update)
                targets_update[i, :len(target_update)] = torch.LongTensor(target_update)

                if self.proof_of_concept and self.verbose:
                    print_list("S-UPDATE", source_update)
                    print_list("T-UPDATE", target_update)

            #

            descendents = tree_src.ravel()

            if self.training and getattr(self, "mask_rate", 0) > 0:
                total_masked = int(self.mask_rate * (len(descendents) - 1))
                masked_nodes = np.random.choice(descendents[1:], size=total_masked, replace=False)
                for each_node in masked_nodes:
                    if getattr(self, "mask_fill", False):
                        each_node.iv = np.random.randint(8, 90)
                    else:
                        each_node.iv = 2

            insert_count = len(descendents) // 2
            if getattr(self, "min_insert", 0) > 0:
                insert_count = max(self.min_insert, insert_count)

            for _ in range(insert_count):
                n: TreeNode = np.random.choice(descendents)
                insert_i, insert_j = 0, 0
                if len(n.children) > 0:
                    insert_i = np.random.randint(0, len(n.children) + 1)
                    insert_j = np.random.randint(0, len(n.children) + 1)
                if getattr(self, "mask_fill", False):
                    n.insert(np.random.randint(8, 90), i=insert_i, j=insert_j)
                else:
                    n.insert(2, i=insert_i, j=insert_j)

            _, mapping = compute_ted(tree_src, tree_dst)

            ## 

            for node_src, node_dst in mapping:
                if node_dst is None:  # DELETE
                    node_src.align = None
                elif node_src is None:  # INSERT
                    node_dst.align = None
                else:  # UPDATE
                    node_src.align = proxy(node_dst)
                    node_dst.align = proxy(node_src)

            sources_insdel_list.append(build_list(tree_src, make_default))
            targets_insdel_list.append(build_list(tree_src, make_delete_label))

            if self.training and getattr(self, "balanced", False):
                if self.proof_of_concept and self.verbose:
                    print_list("S-INSDEL", sources_insdel_list[-1])
                    print_list("T-INSDEL", targets_insdel_list[-1])
                    td = targets_insdel_list[-1]
                    print(td.count(0), td.count(1))
                td = targets_insdel_list[-1]
                td_cnt_0, td_cnt_1 = td.count(0), td.count(1)
                if td_cnt_0 > td_cnt_1:
                    td = np.array(td)
                    pp = np.random.choice(np.where(td == 0)[0], td_cnt_0 - td_cnt_1, replace=False) 
                    # print(pp)
                    td[pp] = -1
                    # print(td)
                    targets_insdel_list[-1] = td.tolist()
                    # assert False

            if self.proof_of_concept and self.verbose:
                print_list("S-INSDEL", sources_insdel_list[-1])
                print_list("T-INSDEL", targets_insdel_list[-1])
                td = targets_insdel_list[-1]
                print(td.count(0), td.count(1))
                # assert False

            ##

            for node_src, node_dst in mapping:
                if node_dst is None:  # DELETE
                    node_src.delete()
                elif node_src is None:  # INSERT
                    node_dst.delete()

            sources_insupd_list.append(build_list(tree_src, make_default))
            targets_insupd_list.append(build_list(tree_dst, make_default_label))

            if self.proof_of_concept and self.verbose:
                print_list("S-INSUPD", sources_insupd_list[-1])
                print_list("T-INSUPD", targets_insupd_list[-1])
            
        #

        max_insdel = targets.size(1)
        max_insdel = max(max_insdel, *[len(each) for each in sources_insdel_list])
        max_insdel = max(max_insdel, *[len(each) for each in targets_insdel_list])
        
        max_insupd = targets.size(1)
        max_insupd = max(max_insupd, *[len(each) for each in sources_insupd_list])
        max_insupd = max(max_insupd, *[len(each) for each in targets_insupd_list])

        sources_insdel = torch.zeros((targets.size(0), max_insdel)).long()
        targets_insdel = torch.full_like(sources_insdel, -1)
        sources_insupd = torch.zeros((targets.size(0), max_insupd)).long()
        targets_insupd = torch.full_like(sources_insupd, -1)

        for i, each in enumerate(sources_insdel_list):
            sources_insdel[i, :len(each)] = torch.LongTensor(each)

        for i, each in enumerate(targets_insdel_list):
            targets_insdel[i, :len(each)] = torch.LongTensor(each)

        for i, each in enumerate(sources_insupd_list):
            sources_insupd[i, :len(each)] = torch.LongTensor(each)

        for i, each in enumerate(targets_insupd_list):
            targets_insupd[i, :len(each)] = torch.LongTensor(each)

        #

        if not half_train:
            sources_delete = sources_delete.to(images.device)
            targets_delete = targets_delete.to(images.device)
            sources_update = sources_update.to(images.device)
            targets_update = targets_update.to(images.device)

        sources_insdel = sources_insdel.to(images.device)
        targets_insdel = targets_insdel.to(images.device)
        sources_insupd = sources_insupd.to(images.device)
        targets_insupd = targets_insupd.to(images.device)

        #

        r = {}

        # 

        memory, pos = self.bottleneck.encode(images)

        #

        if not half_train:
            self.forward_decode_head(r, "delete", self.delete_head, self.criterion_delete, memory, pos, sources_delete, targets_delete)
            self.forward_decode_head(r, "update", self.update_head, self.criterion_update, memory, pos, sources_update, targets_update)
        self.forward_decode_head(r, "insdel", self.delete_head, self.criterion_delete, memory, pos, sources_insdel, targets_insdel)
        self.forward_decode_head(r, "insupd", self.update_head, self.criterion_update, memory, pos, sources_insupd, targets_insupd)

        #

        if half_train:
            r["loss"] = r["loss/insdel"] + r["loss/insupd"]
        else:
            r["loss"] = r["loss/delete"] + r["loss/update"] + r["loss/insdel"] + r["loss/insupd"]
        return r
    
    def forward_decode_head(self, r, name, head, criterion, memory, pos, sources, targets):
        if not getattr(self, "x2", False):
            outputs = self.bottleneck.decode(memory, pos, sources)
        else:
            if name in ["delete", "insdel"]:
                outputs = self.bottleneck.decode(memory, pos, sources)
            else:
                outputs = self.bottleneck_x2.decode(memory, pos, sources)
                
        # print("outputs:", outputs.shape)
        predict = head(outputs)

        predict_view = predict.view(-1, predict.size(-1))
        targets_view = targets.view(-1)
        r[f"loss/{name}"] = criterion(predict_view, targets_view)

        targets_mask = (targets_view != -1)
        r[f"predict/{name}"] = predict_view[targets_mask].detach()
        r[f"targets/{name}"] = targets_view[targets_mask].detach()


# def generate_square_subsequent_mask(sz, device='cpu'):
#     mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask

def generate_square_subsequent_mask(sz, device='cpu'):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
    

def create_mask(cap):
    tgt_seq_len = cap.shape[1]

    cap_mask = generate_square_subsequent_mask(tgt_seq_len, cap.device)
    cap_padding_mask = (cap == 0)
    
    return cap_mask, cap_padding_mask