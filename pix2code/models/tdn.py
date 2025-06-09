from typing import *

import zss
import h5py
import numpy as np
import math
import copy
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
    

def retrieve_fn_old(query, index_path, image_path, code_path: str):
    index = faiss.read_index(index_path)
    _, I = index.search(query, k=1)
    del index
    docids = I[:, 0]  # .ravel()

    r = {}

    if code_path.endswith(".npy"):
        codes = np.load(code_path, "r")
        r["code_embs"] = torch.Tensor(codes[docids])
    else:
        with h5py.File(code_path, "r") as h:
            codes = []
            for docid in docids:
                codes.append(torch.LongTensor(h["ivs"][docid]))
            r["codes"] = torch.stack(codes, dim=0)
    
    images = np.load(image_path, "r")
    r["image_embs"] = torch.Tensor(images[docids])
    del images
    return r


#

global_index = None
global_codes = None
global_code_embs = None
global_images = None

def retrieve_fn(query, index_path, image_path, code_path: str):
    global global_index, global_codes, global_code_embs, global_images

    if global_index is None:
        global_index = faiss.read_index(index_path)
        if code_path.endswith(".npy"):
            global_code_embs = np.load(code_path, "r")
        else:
            with h5py.File(code_path, "r") as h:
                global_codes = h["ivs"][:]
        global_images = np.load(image_path, "r")

    _, I = global_index.search(query, k=1)
    docids = I[:, 0]  # .ravel()

    r = {}

    if global_code_embs is not None:
        r["code_embs"] = torch.Tensor(global_code_embs[docids])
    else:
        r["codes"] = torch.LongTensor(global_codes[docids])
    
    r["image_embs"] = torch.Tensor(global_images[docids])
    return r


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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_head, batch_first=True)
        self.txt_encoder = nn.TransformerEncoder(encoder_layer, num_layer)


        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=num_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layer)

        self.tok_emb = TokenEmbedding(vocab_size, dim)
        self.positional_encoding = PositionalEncoding(dim, dropout=dropout)

    def forward(self, images, inputs):

        cap_mask, cap_padding_mask = create_mask(inputs)


class TreeEditNet(nn.Module):

    def __init__(self, 
                 vocab_size,
                 max_len,

                 backbone, 
                 generator,
                 
                 image_size=256, 
                 patch_size=16, 
                 dim=512, 
                 num_layer=6,
                 num_head=8, 
                 mlp_dim=1024, 
                 dropout=0.1, 
                 emb_dropout=0.1,
                 proof_of_concept: bool = False):
        
        super().__init__()
        self.proof_of_concept = proof_of_concept

        self.backbone = backbone
        self.backbone.eval()
        self.generator = generator

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
        self.insert_head = nn.Linear(dim, 3)
        self.update_head = nn.Linear(dim, vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.fusing = nn.Linear(dim * 2, dim)
    
    def forward(self, batch):
        
        #

        with torch.no_grad():
            predicts, _, _ = self.generator.search(self.backbone, batch)
            predicts = predicts.detach().cpu()

        #

        image = batch["image"]
        targets = batch["code"].long()
        targets_lens = batch["code_len"]

        tree_src = TreeNode.build_tree()
        tree_dst = 


        cap_mask, cap_padding_mask = create_mask(tgt_input)

        memory = self.img_encoder(img)
        # print(memory.shape)  # (batch_size, 257, 512)
        
        batch_size = img.size(0)

        ret_memory = rearrange(memory, "b s d -> (b s) d")
        # ret_memory = memory[:, 0, :]

        r = self.retrieve_fn(ret_memory.detach().cpu().numpy())
        # embb (batch_size * 257, 512)
        # code (batch_size * 257, 356)
        # print(r["image_embs"].shape, r["code_embs"].shape)
        # code = code.to(memory.device)

        if self.has_encoder:
            code = rearrange(code, "(b s) d -> b s d", b=batch_size)
            inp_enc_all = torch.zeros((batch_size, 257, 512), dtype=torch.float32).to(memory.device)
            for i in range(257):
                batched_code = code[:, i, :].to(memory.device)
                # print(batched_code.shape)
                inp_emb = self.positional_encoding(self.tok_emb(batched_code))
                # print(inp_emb.shape)
                inp_enc = self.txt_encoder(inp_emb, src_key_padding_mask=(batched_code == 0))
                inp_enc_all[:, i, :] = inp_enc[:, -1, :]
                del batched_code
                del inp_emb
                del inp_enc
            print("inp_enc_all:", inp_enc_all.shape)
        # else:
            
        # image_embs = r["image_embs"].to(ret_memory.device)
        # image_embs = rearrange(image_embs, "(b s) d -> b s d", b=batch_size)
        # doc_scores = torch.bmm(memory, image_embs.transpose(1, 2))
        # print(doc_scores.shape)  # (batch_size, 257, 257)

        code_embs = r["code_embs"].to(ret_memory.device)
        code_embs = rearrange(code_embs, "(b s) d -> b s d", b=batch_size)
        fusing_memory = self.fusing(torch.cat([memory, code_embs], dim=-1))

        cap_emb = self.positional_encoding(self.tok_emb(tgt_input))
        outs = self.decoder(cap_emb, fusing_memory, tgt_mask = cap_mask, 
                            tgt_key_padding_mask = cap_padding_mask)

        outputs = self.generator(outs)  # (batch, seq_length, num_classes)

        tgt_out = captions[:, 1:]
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), tgt_out.reshape(-1))

        decode_lengths = (caplens - 1).cpu()
        scores = nn.utils.rnn.pack_padded_sequence(outputs, decode_lengths, batch_first=True, enforce_sorted=False).data
        targets = nn.utils.rnn.pack_padded_sequence(tgt_out, decode_lengths, batch_first=True, enforce_sorted=False).data

        return {
            "loss": loss,
            "scores": torch.nn.functional.softmax(scores, dim=-1), 
            "targets": targets
        }
    
    def encode(self, img):
        return self.img_encoder(img)

    def decode(self, caption, memory, cap_mask):
        return self.decoder(self.positional_encoding(self.tok_emb(caption)), memory,
                            tgt_mask = cap_mask)
    
    def predict_init(self, images):
        memory = self.encoder(images)
        return {
            "memory": memory
        }

    def predict_next(self, inputs, context):

        fusing_memory = self.fusing(torch.cat([memory, code_embs], dim=-1))

        cap_emb = self.positional_encoding(self.tok_emb(inputs))
        outs = self.decoder(cap_emb, fusing_memory)
        scores = self.generator(outs[:, -1, :])  # (batch, seq_length, num_classes)
        predicts = torch.argmax(torch.softmax(scores, dim=-1), dim=-1)
        return predicts, scores, {}


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