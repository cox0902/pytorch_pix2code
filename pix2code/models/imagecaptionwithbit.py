from typing import *

from functools import partial

import torch
import torch.nn as nn
import torchvision
import transformers


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, resnet = None, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        if resnet is None:
            resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        gate = self.sigmoid(self.f_beta(decoder_hidden))
        attention_weighted_encoding = attention_weighted_encoding * gate

        return attention_weighted_encoding, alpha


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, max_len, emb_dim), requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(torch.log(torch.tensor(10000.0)) / emb_dim))
        self.embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.embedding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.embedding[:, :x.size(1)]
    

class Rnn(nn.Module):

    def __init__(self, 
                 embedding,
                 attention,
                 embed_dim,
                 encoder_dim,
                 decoder_dim,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding
        self.attention = attention

        self.rnn_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_state(self, x):
        x = x.mean(dim=1)
        h = self.init_h(x)
        c = self.init_c(x)
        return h, c
    
    def forward(self, x, y, hidden):
        x_emb = self.embedding(x)
        y_hat, _ = self.attention(y, hidden[0])
        return self.rnn_step(torch.cat([x_emb[None], y_hat], dim=1), hidden)


class TreeNode:
    
    def __init__(self, iv: Union[int, torch.Tensor], parent: "TreeNode" = None, device = None):
        self.id = hex(id(self))
        self.iv = torch.tensor(iv) if type(iv) == int else iv.clone()
        if device is not None:
            self.iv.to(device)
        elif self.parent is not None:
            self.iv.to(self.parent.iv.device)
        self.parent = parent
        self.children: List[TreeNode] = []
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.hidden_v = None
        self.hidden_h = None

    def add_child(self, iv) -> "TreeNode":
        node = TreeNode(iv, self)
        self.children.append(node)
        return node
    
    def preorder_walk(self, walk):
        walk(self)
        for each in self.children:
            each.preorder_walk(walk)

    def preorder_walk_children(self, walk):
        for each in self.children:
            each.preorder_walk(walk)

    @staticmethod
    def _build_list(n: "TreeNode", ivs: List):
        ivs.append(n.iv)
        if len(n.children) > 0:
            ivs.append(5)  # [LB]
            for each in n.children:
                TreeNode._build_list(each, ivs)
            ivs.append(6)  # [RB]
            
    def build_list(self) -> List[int]:
        ivs = [3]  # [START]
        for each in self.children:
            TreeNode._build_list(each, ivs)
        ivs.append(4)  # [END]
        return ivs

    @staticmethod
    def _make_graph(n: "TreeNode", vocabs: List[str], dot: "Digraph"):
        label = vocabs[n.iv]
        # if n.left is not None:
        #     label = f"{vocabs[n.left.iv]} < " + label
        # if n.right is not None:
        #     label = label + f" > {vocabs[n.right.iv]}"
        dot.node(name=n.id, label=label)
        if n.parent is not None:
            dot.edge(n.parent.id, n.id)

    def visualize(self, vocabs: List[str]) -> "Digraph":
        dot = Digraph()
        self.preorder_walk(partial(TreeNode._make_graph, vocabs=vocabs, dot=dot))
        return dot
    
    @staticmethod
    def _train(n: "TreeNode", prds: List, tgts: List, fc, rnn_h: Rnn, rnn_v: Rnn, y):
        # print("=" * 100)
        if n.left.iv != 3:
            n.hidden_v = n.left.hidden_v
            # print(rnn_v.name, [vocabs[each] for each in n.hidden_v])
        else:
            n.hidden_v = rnn_v.forward(n.parent.iv, y, n.parent.hidden_v)
        n.hidden_h = rnn_h.forward(n.left.iv, y, n.left.hidden_h)

        # print("=>", vocabs[n.iv])
        prds.append(fc(n.hidden_h[0], n.hidden_v[0]))
        tgts.append(n.iv)

        for each in n.children[1:]:
            TreeNode._train(each, prds, tgts, fc=fc, rnn_h=rnn_h, rnn_v=rnn_v, y=y)

        # if n.right.iv == 4:
        #     print("=" * 100)
        #     print(rnn_v.name, [vocabs[each] for each in n.hidden_v])
        #     rnn_h.forward(n.iv, n.hidden_h)
        #     print("=>", vocabs[n.right.iv])

    @staticmethod
    def _assign_init_hidden_state(n: "TreeNode", hidden_h):
        if n.iv == 3:
            n.hidden_h = hidden_h
    
    def train(self, fc, rnn_h: Rnn, rnn_v: Rnn, y):

        prds, tgts = [], []

        # self.hidden_h = rnn_h.init_hidden_state()
        # self.hidden_v = rnn_v.forward(self.iv, rnn_v.init_hidden_state())
        self.hidden_v = rnn_v.init_hidden_state(y)

        # init_left.hidden_h = self.hidden_h[:]  # self.hidden_h.clone()
        # init_left.hidden_v = self.hidden_v[:]  # self.hidden_v.clone()
        # hidden_h = rnn_h.forward(self.iv, rnn_h.init_hidden_state())
        hidden_h = rnn_h.init_hidden_state(y)
        self.preorder_walk_children(partial(TreeNode._assign_init_hidden_state, hidden_h=hidden_h))
    
        for each in self.children[1:]:
            TreeNode._train(each, prds, tgts, fc=fc, rnn_h=rnn_h, rnn_v=rnn_v, y=y)
        # self.preorder_walk_children(partial(TreeNode._train, rnn_h=rnn_h, rnn_v=rnn_v))
        return prds, tgts

    @staticmethod
    def _finalize(n: "TreeNode"):
        n.children = [TreeNode(3, n)] + n.children + [TreeNode(4, n)]
        n.children[0].right = n.children[1]
        n.children[-1].left = n.children[-2]
        for i in range(1, len(n.children) - 1):
            n.children[i].left = n.children[i - 1]
            n.children[i].right = n.children[i + 1]
            TreeNode._finalize(n.children[i])

    @staticmethod
    def build_tree(code, device):
        root = TreeNode(3, device=device)
        node = root
        queue: List[TreeNode] = [node]
        for iv in code:
            if iv == 3:  # [START]
                continue
            if iv == 4:  # [END]
                break
            if iv == 5:  # [LB]
                queue.append(node)
                continue
            if iv == 6:  # [RB]
                queue.pop()
                continue
            assert iv != 0
            node = queue[-1].add_child(iv)

        TreeNode._finalize(root)
        return root


class Fusing(nn.Module):

    def __init__(self, decoder_dim, vocab_size, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(decoder_dim * 2, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()
    
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, y):
        cat = torch.cat([x, y], dim=1)
        return self.fc(self.dropout(cat))


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, max_len, attention_dim, embed_dim, decoder_dim, vocab_size, 
                 encoder_dim=2048, dropout=0.5, pos_embed=None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        if pos_embed == '1':
            self.pos_embedding = PositionalEmbedding(max_len, embed_dim)
        else:
            self.pos_embedding = None

        self.rnn_h = Rnn(self.embedding, self.attention, embed_dim, encoder_dim, decoder_dim)
        self.rnn_v = Rnn(self.embedding, self.attention, embed_dim, encoder_dim, decoder_dim)

        # self.decode_step_h = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # self.decode_step_v = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # self.init_h_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        # self.init_h_v = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c_v = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell

        self.fc = Fusing(decoder_dim, vocab_size, dropout=dropout)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, encoded_captions, caption_lengths, return_alphas = False):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        # Embedding
        # embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # if self.pos_embedding is not None:
        #     embeddings = self.pos_embedding(embeddings)

        predict = []
        targets = []
        if return_alphas:
            alphas = []

        for bi in range(batch_size):
            # print(bi)

            alphas_nb = []

            tree = TreeNode.build_tree(encoded_captions[bi, :caption_lengths[bi]], device=encoder_out.device)
            prd, tgt = tree.train(self.fc, self.rnn_h, self.rnn_v, encoder_out[bi][None])

            # print(len(prd), prd[0].shape)
            # print(tgt)

            predict.extend(prd)
            targets.extend(tgt)

            if return_alphas:
                alphas.append(torch.stack(alphas_nb).sum(dim=0).squeeze())

        if return_alphas:
            return torch.stack(predict), torch.stack(targets), torch.stack(alphas)
        return torch.cat(predict, dim=0), torch.stack(targets)
    
    def predict(self, encoder_out, captions, hiddens = None):
        # Embedding
        # print(captions.shape)
        embeddings = self.embedding(captions)  # (batch_size, embed_dim)
        # print(embeddings.shape)

        # Initialize LSTM state
        if hiddens is None:
            h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

            decode_length = captions.size(-1)

            for t in range(decode_length):
                attention_weighted_encoding, alpha = self.attention(encoder_out, h)
                gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1),
                    (h, c))  # (batch_size_t, decoder_dim)
                preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
        else:
            h, c = hiddens
            # print("h, c:", h.shape, c.shape)

            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

        return preds, alpha, (h, c)


class ImageCaptionWithBit(nn.Module):
    """
    Image captioning with bi-directional tree rnn.
    """

    def __init__(self, resnet, vocab_size: int, max_len,                  
                 proof_of_concept: bool = False):
        super().__init__()
        self.proof_of_concept: bool = proof_of_concept
        self.vocab_size = vocab_size
        self.alpha_c = 1.
        self.encoder = Encoder(resnet)
        self.decoder = DecoderWithAttention(max_len,
                                            attention_dim=512,
                                            embed_dim=512,
                                            decoder_dim=512,
                                            vocab_size=vocab_size,
                                            dropout=0.2)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, batch):
        imgs = batch["image"]
        caps = batch["code"].long()
        caplens = batch["code_len"]

        # Forward prop.
        imgs = self.encoder(imgs)
        scores, targets = self.decoder(imgs, caps, caplens, return_alphas=False)
        # scores, targets, alphas = self.decoder(imgs, caps, caplens)

        # print(scores.shape, targets.shape)
        # print(alphas.shape)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        # targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # scores = nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        # targets = nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = self.criterion(scores, targets)

        # Add doubly stochastic attention regularization
        # loss += self.alpha_c * ((1. - alphas) ** 2).mean()

        return {
            "loss": loss, 
            "scores": torch.nn.functional.softmax(scores, dim=-1), 
            "targets": targets
        }
    
    # def predict(self, batch, hiddens = None):
    #     imgs = batch["image"]
    #     caps = batch["code"].long()

    #     imgs = self.encoder(imgs)

    #     preds, _, hiddens = self.decoder.predict(imgs, caps, hiddens)

    #     return {
    #         "scores": preds,
    #         "hiddens": hiddens
    #     }

    def predict_init(self, images):
        batch_size = images.size(0)
        
        # with torch.no_grad() should be called outside this scope.
        encoder_out = self.encoder(images)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        h, c = self.decoder.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        return {
            "encoder_out": encoder_out,
            "h": h,
            "c": c,
        }

    def predict_next(self, inputs, contexts):
        
        # with torch.no_grad() should be called outside this scope.
        scores, _, (h, c) = self.decoder.predict(contexts["encoder_out"], inputs, (contexts["h"], contexts["c"]))
        predicts = torch.argmax(torch.softmax(scores, dim=-1), dim=-1)
        
        return predicts, scores, {
            "h": h,
            "c": c,
        }
