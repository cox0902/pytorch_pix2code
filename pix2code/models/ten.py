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

from ..tree import TreeNode


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SPT(nn.Module):
    def __init__(self, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim,
                 channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # print(x.shape)  # (32, 256, 512)
 
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size) 
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, dim, num_head, num_layers, norm=None):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_head, batch_first = True)
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None):
        
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
#


def _mask_pads(target, ll, smooth_obj):
    pad_mask = target.eq(0)
    if pad_mask.any():
        ll.masked_fill_(pad_mask, 0.0)
        smooth_obj.masked_fill_(pad_mask, 0.0)
    return ll.squeeze(-1), smooth_obj.squeeze(-1)


class BottleNeck(nn.Module):

    def __init__(self, 
                 vocab_size,
                 image_size=256, 
                 patch_size=16, 
                 dim=512, 
                 num_layer=6,
                 num_head=8, 
                 mlp_dim=1024, 
                 dropout=0.1, 
                 emb_dropout=0.1,):
        super().__init__()

        self.img_encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=num_layer,
            heads=num_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=dim, nhead=num_head, batch_first=True)
        # self.txt_encoder = nn.TransformerEncoder(encoder_layer, num_layer)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=num_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layer)

        self.tok_emb = TokenEmbedding(vocab_size, dim)
        self.positional_encoding = PositionalEncoding(dim, dropout=dropout)

    def forward(self, images, inputs):

        _, cap_padding_mask = create_mask(inputs)

        memory = self.img_encoder(images)

        cap_emb = self.positional_encoding(self.tok_emb(inputs))
        outs = self.decoder(cap_emb, memory, tgt_key_padding_mask = cap_padding_mask)
        return outs
    
    def encode(self, images):
        return self.img_encoder(images)
    
    def decode(self, memory, inputs):
        _, cap_padding_mask = create_mask(inputs)
        cap_emb = self.positional_encoding(self.tok_emb(inputs))
        outs = self.decoder(cap_emb, memory, tgt_key_padding_mask = cap_padding_mask)
        return outs


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


class TreeEditNet(nn.Module):

    def __init__(self, 
                 vocab_size,
                 max_len,

                 backbone=None, 
                 generator=None,
                 
                 image_size=256, 
                 patch_size=16, 
                 dim=512, 
                 num_layer=6,
                 num_head=8, 
                 mlp_dim=1024, 
                 dropout=0.1, 
                 emb_dropout=0.1,

                 mask_rate = None,

                 verbose = None,
                 proof_of_concept: bool = False):
        
        super().__init__()
        self.proof_of_concept = proof_of_concept
        self.verbose = (verbose == "1")

        self.mask_rate = (float(mask_rate) if mask_rate is not None else 0.0)

        print({
            "mask_rate": self.mask_rate,
            "verbose": self.verbose
        })

        self.backbone = backbone
        if backbone is not None:
            self.backbone.eval()
            self.generator = generator(max_seq_len=max_len)

        self.bottleneck = BottleNeck(
            vocab_size,
            image_size=image_size, 
            patch_size=patch_size, 
            dim=dim, 
            num_layer=num_layer,
            num_head=num_head, 
            mlp_dim=mlp_dim, 
            dropout=dropout, 
            emb_dropout=emb_dropout,
        )

        self.delete_head = nn.Linear(dim, 2)
        # self.insert_head = nn.Linear(dim, 3)
        self.update_head = nn.Linear(dim, vocab_size)

        self.criterion_delete = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_update = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch):
        # if self.training:
        return self.forward_training(batch)
        # else:
        #     return self.forward_inference(batch)
        
    def forward_training(self, batch):

        #

        images = batch["image"]
        targets = batch["code"].long()  # (B, S)
        # targets_lens = batch["code_len"]
        batch_size = targets.size(0)

        #

        # self.backbone.to(images.device)
        if self.backbone is not None:
            with torch.no_grad():
                predicts, _, _ = self.generator.search(self.backbone, batch)
                predicts = predicts.detach().cpu()
        else:
            predicts = batch["pred"]

        #

        sources_delete = torch.zeros((batch_size, MAX_SEQ_LEN), dtype=torch.long)
        targets_delete = torch.full_like(sources_delete, -1)

        sources_update = torch.zeros_like(sources_delete)
        targets_update = torch.full_like(sources_delete, -1)

        sources_insdel_list = []
        targets_insdel_list = []
        sources_insupd_list = []
        targets_insupd_list = []

        for i, (src, dst) in enumerate(zip(predicts, targets)):

            tree_src = TreeNode.build_tree(src)
            tree_dst = TreeNode.build_tree(dst)

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

            sources_delete[i, :len(source_delete)] = torch.LongTensor(source_delete)
            targets_delete[i, :len(target_delete)] = torch.LongTensor(target_delete)

            if self.proof_of_concept and self.verbose:
                print("-" * 80)
                print_list("S-DELETE", sources_delete[i], ignore_idx=0)
                print_list("T-DELETE", target_delete)

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

            if getattr(self, "mask_rate", 0) > 0:
                total_masked = int(self.mask_rate * (len(descendents) - 1))
                masked_nodes = np.random.choice(descendents[1:], size=total_masked, replace=False)
                for each_node in masked_nodes:
                    each_node.iv = 5

            for _ in range(len(descendents) // 2):
                n: TreeNode = np.random.choice(descendents)
                insert_i, insert_j = 0, 0
                if len(n.children) > 0:
                    insert_i = np.random.randint(0, len(n.children) + 1)
                    insert_j = np.random.randint(0, len(n.children) + 1)
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

            if self.proof_of_concept and self.verbose:
                print_list("S-INSDEL", sources_insdel_list[-1])
                print_list("T-INSDEL", targets_insdel_list[-1])

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

        memory = self.bottleneck.encode(images)

        #

        outputs = self.bottleneck.decode(memory, sources_delete)
        predict_delete = self.delete_head(outputs)

        predict_delete_view = predict_delete.view(-1, predict_delete.size(-1))
        targets_delete_view = targets_delete.view(-1)
        r["loss/delete"] = self.criterion_delete(predict_delete_view, targets_delete_view)

        targets_delete_mask = (targets_delete_view != -1)
        r["predict/delete"] = predict_delete_view[targets_delete_mask].detach()
        r["targets/delete"] = targets_delete_view[targets_delete_mask].detach()

        #

        outputs = self.bottleneck.decode(memory, sources_update)
        predict_update = self.update_head(outputs)

        predict_update_view = predict_update.view(-1, predict_update.size(-1))
        targets_update_view = targets_update.view(-1)
        r["loss/update"] = self.criterion_update(predict_update_view, targets_update_view)

        targets_update_mask = (targets_update_view != -1)
        r["predict/update"] = predict_update_view[targets_update_mask].detach()
        r["targets/update"] = targets_update_view[targets_update_mask].detach()

        #

        outputs = self.bottleneck.decode(memory, sources_insdel)
        predict_insdel = self.delete_head(outputs)

        predict_insdel_view = predict_insdel.view(-1, predict_insdel.size(-1))
        targets_insdel_view = targets_insdel.view(-1)
        r["loss/insdel"] = self.criterion_delete(predict_insdel_view, targets_insdel_view)

        targets_insdel_mask = (targets_insdel_view != -1)
        r["predict/insdel"] = predict_insdel_view[targets_insdel_mask].detach()
        r["targets/insdel"] = targets_insdel_view[targets_insdel_mask].detach()
        
        #

        outputs = self.bottleneck.decode(memory, sources_insupd)
        predict_insupd = self.update_head(outputs)

        predict_insupd_view = predict_insupd.view(-1, predict_insupd.size(-1))
        targets_insupd_view = targets_insupd.view(-1)
        r["loss/insupd"] = self.criterion_update(predict_insupd_view, targets_insupd_view)

        targets_insupd_mask = (targets_insupd_view != -1)
        r["predict/insupd"] = predict_insupd_view[targets_insupd_mask]
        r["targets/insupd"] = targets_insupd_view[targets_insupd_mask]

        #

        r["loss"] = r["loss/delete"] + r["loss/update"] + r["loss/insdel"] + r["loss/insupd"]
        return r
    
    def forward_inference(self, batch):

        #

        images = batch["image"]
        targets = batch["code"].long()  # (B, S)
        # targets_lens = batch["code_len"]
        batch_size = targets.size(0)

        #

        # self.backbone.to(images.device)
        with torch.no_grad():
            predicts, _, _ = self.generator.search(self.backbone, batch)
            predicts = predicts.detach().cpu()

        for i, (src, dst) in enumerate(zip(predicts, targets)):
        
            tree_src = TreeNode.build_tree(src)
            source_delete = build_list(tree_src, make_default)
            sources_delete = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.long)
            sources_delete[0, :len(source_delete)] = torch.LongTensor(source_delete)

            with torch.no_grad():
                outputs = self.bottleneck(images, sources_delete)
                predict_delete = self.delete_head(outputs)
            
            tree_del = TreeNode.build_tree(sources_delete[0], predict_delete[0])

        return {}
    

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