import faiss
import h5py
import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
    

def retrieve_fn(query, index_path, image_path, code_path, top_most: bool = False):
    index = faiss.read_index(index_path)
    _, I = index.search(query, k=1 if top_most else 2)
    del index
    if not top_most:
        docids = I[:, 1]
    docids = I[:, 0]
    with h5py.File(code_path, "r") as h:
        sorted_docids = np.sort(docids)
        sorted_indice = np.argsort(docids)
        print(sorted_docids)
        codes = h["ivs"][sorted_docids]
        codes = codes[sorted_indice]
    images = np.load(image_path, "r")
    features = images[docids]
    del images
    return features, codes


class Rag2Code(nn.Module):

    def __init__(self, 
                 vocab_size, 
                 max_len,
                 retrieve_fn,
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

        self.retrieve_fn = retrieve_fn  # Retriever(retriever_index_path, retriever_database)

        self.img_encoder = ViT(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = num_layer,
            heads = num_head,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout
        )
        # self.encoder = torchvision.models.vit_b_16()

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_head, batch_first=True)
        self.txt_encoder = nn.TransformerEncoder(encoder_layer, num_layer)

        # self.decoder = Decoder(
        #     dim = dim, 
        #     num_head = num_head, 
        #     num_layers = num_layer
        # )

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layer)

        self.tok_emb = TokenEmbedding(vocab_size, dim)
        self.positional_encoding = PositionalEncoding(dim, dropout=dropout)
        self.generator = nn.Linear(dim, vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, batch):
        img = batch["image"]
        captions = batch["code"].long()
        caplens = batch["code_len"]

        tgt_input = captions[:, :-1]

        cap_mask, cap_padding_mask = create_mask(tgt_input)

        memory = self.img_encoder(img)
        # print(memory.shape)  # (batch_size, 257, 512)

        ret_memory = rearrange(memory, "b s d -> (b s) d")

        embb, code = self.retrieve_fn(ret_memory.detach().cpu().numpy())
        # embb (257, 512)
        # code 
        print(embb.shape, code.shape)
        code = torch.LongTensor(code).to(ret_memory.device)

        inp_emb = self.positional_encoding(self.tok_emb(code))
        print(inp_emb.shape)

        inp_enc = self.txt_encoder(inp_emb)
        print(inp_enc.shape)

        doc_scores = torch.bmm(ret_memory, embb.transpose(0, 1))
        print(doc_scores.shape)

        cap_emb = self.positional_encoding(self.tok_emb(tgt_input),
                                           src_key_padding_mask=(code == 0))
        outs = self.decoder(cap_emb, memory, tgt_mask = cap_mask, 
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