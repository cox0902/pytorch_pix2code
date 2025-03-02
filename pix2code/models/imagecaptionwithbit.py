from typing import *

from functools import partial

# import zss
import functools
import collections
import itertools
import copy
# from apted import APTED, Config
# from networkx import DiGraph, optimize_graph_edit_distance

import numpy as np
import torch
import torch.nn as nn
import torchvision
from einops import rearrange, repeat, reduce
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor


class FpnEncoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, resnet):
        super().__init__()
        # Extract 4 main layers
        self.body = create_feature_extractor(
            resnet, return_nodes={f'layer{k}': str(v) for v, k in enumerate([1, 2, 3, 4])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels=self.out_channels)

    def forward(self, images):
        x = self.body(images)
        x = self.fpn(x)
        return [x[str(i)] for i in range(4)]


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


class ChannelAttention(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, hidden):
        pass


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
        encoder_out = rearrange(encoder_out, "B W H C -> B (W H) C")
        # print(encoder_out.shape, decoder_hidden.shape)

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # print(att1.shape, att2.shape)
        
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        gate = self.sigmoid(self.f_beta(decoder_hidden))
        attention_weighted_encoding = attention_weighted_encoding * gate

        return attention_weighted_encoding, alpha


class AttentionSum(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, encoder_out, decoder_hidden):
        encoder_out = rearrange(encoder_out, "B W H C -> B (W H) C")
        return encoder_out.sum(dim=1), None
    

class AttentionAvg(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, encoder_out: torch.Tensor, decoder_hidden):
        encoder_out = rearrange(encoder_out, "B W H C -> B (W H) C")
        return encoder_out.mean(dim=1), None


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
                 conditional_init: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional_init = conditional_init
        self.embedding = embedding
        self.attention = attention

        self.rnn_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        if not conditional_init:
            self.init_h = nn.Linear(encoder_dim, decoder_dim)
            self.init_c = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.init_h = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
            self.init_c = nn.Linear(encoder_dim + decoder_dim, decoder_dim)

    def init_hidden_state(self, x, y = None):
        x = reduce(x, "B W H C -> B C", reduction="mean")   # x.mean(dim=1)
        if y is not None:
            x = torch.cat([x, y], dim=-1)
        h = self.init_h(x)
        c = self.init_c(x)
        return h, c
    
    def forward(self, x, y, hidden):
        x_emb = self.embedding(x)
        if self.attention is not None:
            y_hat, alpha = self.attention(y, hidden[0])
        else:
            # print(y.shape)
            y_hat = reduce(y, "B W H C -> B C", reduction="sum")
            alpha = None
        # print(y_hat.shape, x_emb.shape, x)
        return self.rnn_step(torch.cat([x_emb[None], y_hat], dim=1), hidden), alpha



vocabs = "啊啋啌啍啎問啐啑啒啓啔啕啖啗啘啙啚啛啜啝啞啟啠啡啢啣啤啥啦啧啨啩啪啫啬啭啮啯啰啱啲啳啴啵啶啷啸啹啺啻啼啽啾啿喀喁喂喃善喅喆喇喈喉喊喋喌喍喎喏喐喑喒喓喔喕喖喗喘喙喚喛喜喝喞喟喠喡喢喣"

class Profile(object):
    """
        Represents a PQ-Gram Profile, which is a list of PQ-Grams. Each PQ-Gram is represented by a
        deque. This class relies on the tree.Node classe.
    """

    def __init__(self, root, p=2, q=3):
        """
            Builds the PQ-Gram Profile of the given tree, using the p and q parameters specified.
            The p and q parameters do not need to be specified, however, different values will have
            an effect on the distribution of the calculated edit distance. In general, smaller values
            of p and q are better, though a value of (1, 1) is not recommended, and anything lower is
            invalid.
        """
        super(Profile, self).__init__()
        ancestors = collections.deque('*'*p, maxlen=p)
        self.list = list()

        self.profile(root, p, q, ancestors)
        self.sort()

    def profile(self, root, p, q, ancestors):
        """
            Recursively builds the PQ-Gram profile of the given subtree. This method should not be called
            directly and is called from __init__.
        """
        ancestors.append(vocabs[root.iv])
        siblings = collections.deque('*'*q, maxlen=q)

        if(len(root.children) == 0):
            self.append(itertools.chain(ancestors, siblings))
        else:
            for child in root.children:
                siblings.append(vocabs[child.iv])
                self.append(itertools.chain(ancestors, siblings))
                self.profile(child, p, q, copy.copy(ancestors))
            for i in range(q-1):
                siblings.append("*")
                self.append(itertools.chain(ancestors, siblings))

    def edit_distance(self, other):
        """
            Computes the edit distance between two PQ-Gram Profiles. This value should always
            be between 0.0 and 1.0. This calculation is reliant on the intersection method.
        """
        union = len(self) + len(other)
        return 1.0 - 2.0*(self.intersection(other)/union)

    def intersection(self, other):
        """
            Computes the set intersection of two PQ-Gram Profiles and returns the number of
            elements in the intersection.
        """
        intersect = 0.0
        i = j = 0
        maxi = len(self)
        maxj = len(other)
        while i < maxi and j < maxj:
            intersect += self.gram_edit_distance(self[i], other[j])
            if self[i] == other[j]:
                i += 1
                j += 1
            elif self[i] < other[j]:
                i += 1
            else:
                j += 1
        return intersect

    @functools.lru_cache()
    def gram_edit_distance(self, gram1, gram2):
        """
            Computes the edit distance between two different PQ-Grams. If the two PQ-Grams are the same
            then the distance is 1.0, otherwise the distance is 0.0. Changing this will break the
            metrics of the algorithm.
        """
        distance = 0.0
        if gram1 == gram2:
            distance = 1.0
        return distance

    def sort(self):
        """
            Sorts the PQ-Grams by the concatenation of their labels. This step is automatically performed
            when a PQ-Gram Profile is created to ensure the intersection algorithm functions properly and
            efficiently.
        """
        self.list.sort(key=lambda x: ''.join(x))

    def append(self, value):
        self.list.append(tuple(value))

    @functools.lru_cache(maxsize=2)
    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return str(self.list)

    def __str__(self):
        return str(self.list)

    @functools.lru_cache(maxsize=32)
    def __getitem__(self, key):
        return self.list[key]

    def __iter__(self):
        return iter(self.list)

# class CustomConfig(Config):
#     def rename(self, node1, node2):
#         return 1 if node1.iv != node2.iv else 0

#     def children(self, node):
#         return node.children




class TreeNode:
    
    def __init__(self, device, iv: Union[int, torch.Tensor], parent: "TreeNode" = None):
        self.id = hex(id(self))
        self.set_iv(iv, device=device)
        self.iv_ = self.iv

        # print(self.iv, device, self.iv.device)

        self.parent = parent

        self.children: List[TreeNode] = []
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.hidden_v = None
        self.hidden_h = None
        self.height = 1 if parent is None else parent.height + 1
        self.score = 0
        self.count = 1

    def set_iv(self, iv, device):
        self.iv = torch.tensor(iv).to(device) if type(iv) == int else iv.clone().to(device)

    def add_child(self, device, iv) -> "TreeNode":
        node = TreeNode(device, iv, self)
        self.children.append(node)
        return node
    
    def preorder_walk(self, walk):
        walk(self)
        for each in self.children:
            each.preorder_walk(walk)

    def postorder_walk(self, walk):
        for each in self.children:
            each.postorder_walk(walk)
        walk(self)

    def preorder_walk_children(self, walk):
        for each in self.children:
            each.preorder_walk(walk)

    @staticmethod
    def _make_logit(iv, device = "cpu"):
        logit = torch.zeros((90, )).to(device)
        logit[iv] = 1.0
        return logit

    @staticmethod
    def _make_logits(n: "TreeNode", ivs: List, device = "cpu"):
        ivs.append(TreeNode._make_logit(n.iv.item(), device=device))
        if len(n.children) > 2:
            ivs.append(TreeNode._make_logit(5, device=device))  # [LB]
            for each in n.children[1:-1]:
                TreeNode._make_logits(each, ivs, device=device)
            ivs.append(TreeNode._make_logit(6, device=device))  # [RB]

    def make_logits(self, device = None):
        ivs = [TreeNode._make_logit(3, device=device)]  # [START]
        for each in self.children[1:-1]:
            TreeNode._make_logits(each, ivs, device=device)
        ivs.append(TreeNode._make_logit(4, device=device))  # [END]
        return ivs

    @staticmethod
    def _build_list(n: "TreeNode", ivs: List):
        ivs.append(n.iv.item())
        if len(n.children) > 2:
            ivs.append(5)  # [LB]
            for each in n.children[1:-1]:
                TreeNode._build_list(each, ivs)
            ivs.append(6)  # [RB]
            

    def build_list(self) -> List[int]:
        ivs = [3]  # [START]
        for each in self.children[1:-1]:
            TreeNode._build_list(each, ivs)
        ivs.append(4)  # [END]
        return ivs

    def format_label(self, vocabs, show_score: bool = False):
        label = vocabs[self.iv]
        if show_score:
            label += f"\n{self.score:.4f}"
        return label 

    @staticmethod
    def _make_graph(n: "TreeNode", 
                    vocabs: List[str],
                    dot,
                    show_connections: bool = False,
                    show_controls: bool = True, 
                    show_score: bool = False):
        
        if show_controls or (not show_controls and n.iv >= 8):
            dot.node(name=n.id, label=n.format_label(vocabs, show_score=show_score))
        with dot.subgraph() as s:
            s.attr(rank="same")
            children = n.children if show_controls else n.children[1:-1]
            for each in children:
                s.node(name=each.id, label=each.format_label(vocabs, show_score=show_score))
                if show_controls or (not show_controls and n.iv >= 8):
                    dot.edge(n.id, each.id)

        if show_connections:
            children = n.children[:-1] if show_controls else n.children[1:-2]
            for each in children:
                dot.edge(each.id, each.right.id, style="dashed", arrowhead="none", arrowtail="none")

        for each in n.children:
            TreeNode._make_graph(each, vocabs, dot, 
                                 show_connections=show_connections, 
                                 show_controls=show_controls, 
                                 show_score=show_score)

    def visualize(self, 
                  vocabs: List[str], 
                  show_connections: bool = False,
                  show_controls: bool = True, 
                  show_score: bool = False) -> "Digraph":
        from graphviz import Digraph
        dot = Digraph()

        TreeNode._make_graph(self, 
                             vocabs=vocabs, 
                             dot=dot, 
                             show_connections=show_connections,
                             show_controls=show_controls, 
                             show_score=show_score)
        return dot
    

    @staticmethod
    def _to_graph(n: "TreeNode", g: "DiGraph"):
        g.add_node(n.id)
        for each in n.children:
            TreeNode._to_graph(each, g)
            g.add_edge(n.id, each.id)

    def to_graph(self):
        g = DiGraph()
        self._to_graph(self, g)
        return g

    @staticmethod
    def insert_cost(n, cost):
        return cost
    
    @staticmethod
    def remove_cost(n, cost):
        return cost
    
    @staticmethod
    def update_cost(n1, n2, cost):
        return cost if n1.iv != n2.iv else 0
    
    @staticmethod
    def get_children(n: "TreeNode"):
        return n.children

    @staticmethod
    def ted(n1: "TreeNode", n2: "TreeNode"):
        # return zss.distance(n1, n2,
        #                     TreeNode.get_children, 
        #                     insert_cost=partial(TreeNode.insert_cost, cost=1.0),
        #                     remove_cost=partial(TreeNode.remove_cost, cost=1.0),
        #                     update_cost=partial(TreeNode.update_cost, cost=1.0))
        p1 = Profile(n1)
        p2 = Profile(n2)
        return p1.edit_distance(p2)
        # apted = APTED(n1, n2, CustomConfig())
        # return apted.compute_edit_distance()
        # g1 = n1.to_graph()
        # g2 = n2.to_graph()
        # for v in optimize_graph_edit_distance(g1, g2):
        #     minv = v
        # return minv


    @staticmethod
    def _train(n: "TreeNode", prds: List, tgts: List, fc: "Fusing", rnn_h: Rnn, rnn_v: Rnn, y, 
               topos: Optional[List] = None,
               alphas_h: Optional[List] = None,
               alphas_v: Optional[List] = None,
               teacher_forcing_rate = 1.0):
        # print("=" * 100)
        if teacher_forcing_rate == 1.0:
            if n.left.iv != 3:
                n.hidden_v = n.left.hidden_v
                # print(rnn_v.name, [vocabs[each] for each in n.hidden_v])
            else:
                n.hidden_v, alpha_v = rnn_v.forward(n.parent.iv, y, n.parent.hidden_v)
                if alphas_v is not None:
                    alphas_v.append(alpha_v)
        else:
            parent_iv = n.parent.iv if np.random.rand() < teacher_forcing_rate else n.parent.iv_
            n.hidden_v, alpha_v = rnn_v.forward(parent_iv, y, n.parent.hidden_v)
            if alphas_v is not None:
                alphas_v.append(alpha_v)
            
        if n.left.iv == 3:
            if getattr(rnn_h, "conditional_init", False):
                assert n.left.hidden_h is None
                n.left.hidden_h = rnn_h.init_hidden_state(y, n.hidden_v[0])

        left_iv = n.left.iv if np.random.rand() < teacher_forcing_rate else n.left.iv_
        n.hidden_h, alpha_h = rnn_h.forward(left_iv, y, n.left.hidden_h)
        if alphas_h is not None:
            alphas_h.append(alpha_h)

        # print("=>", vocabs[n.iv])
        if topos is not None:
            out, p_h, p_v = fc(n.hidden_h[0], n.hidden_v[0])
            t_h = 0 if n.right.iv == 4 else 1
            t_v = 0 if len(n.children) == 2 else 1
            topos.append(torch.Tensor([p_h, t_h, p_v, t_v]))
        else:
            out = fc(n.hidden_h[0], n.hidden_v[0])

        n.iv_ = torch.argmax(nn.functional.softmax(out, dim=-1), dim=-1)[0]

        prds.append(out)
        tgts.append(n.iv)

        children = n.children[1:-1] if topos is not None else n.children[1:]
        for each in children:
            TreeNode._train(each, prds, tgts, fc=fc, rnn_h=rnn_h, rnn_v=rnn_v, y=y, 
                            topos=topos, alphas_h=alphas_h, alphas_v=alphas_v, 
                            teacher_forcing_rate=teacher_forcing_rate)

        # if n.right.iv == 4:
        #     print("=" * 100)
        #     print(rnn_v.name, [vocabs[each] for each in n.hidden_v])
        #     rnn_h.forward(n.iv, n.hidden_h)
        #     print("=>", vocabs[n.right.iv])

    @staticmethod
    def _assign_init_hidden_state(n: "TreeNode", hidden_h):
        if n.iv == 3:
            n.hidden_h = hidden_h

    @staticmethod
    def _mask_iv_(n: "TreeNode"):
        if n.iv != 3:
            n.iv_ = None
    
    def  train(self, fc, rnn_h: Rnn, rnn_v: Rnn, y, prds: List, tgts: List, 
              topos: Optional[List] = None, alphas_h: Optional[List] = None, alphas_v: Optional[List] = None,
              teacher_forcing_rate = 1.0):

        # self.hidden_h = rnn_h.init_hidden_state()
        # self.hidden_v = rnn_v.forward(self.iv, rnn_v.init_hidden_state())
        self.hidden_v = rnn_v.init_hidden_state(y)

        # init_left.hidden_h = self.hidden_h[:]  # self.hidden_h.clone()
        # init_left.hidden_v = self.hidden_v[:]  # self.hidden_v.clone()
        # hidden_h = rnn_h.forward(self.iv, rnn_h.init_hidden_state())
        if getattr(rnn_h, "conditional_init", False):
            pass
        else:
            hidden_h = rnn_h.init_hidden_state(y)
            self.preorder_walk_children(partial(TreeNode._assign_init_hidden_state, hidden_h=hidden_h))

        children = self.children[1:-1] if topos is not None else self.children[1:]
        for each in children:
            TreeNode._train(each, prds, tgts, fc=fc, rnn_h=rnn_h, rnn_v=rnn_v, y=y, 
                            topos=topos, alphas_h=alphas_h, alphas_v=alphas_v, 
                            teacher_forcing_rate=teacher_forcing_rate)
        # self.preorder_walk_children(partial(TreeNode._train, rnn_h=rnn_h, rnn_v=rnn_v))
    
    @staticmethod
    def _update_score(n: "TreeNode", fc, rnn_h: Rnn, rnn_v: Rnn, y, has_topo: bool = False):
        # print("=" * 100)
        if n.left.iv != 3:
            n.hidden_v = n.left.hidden_v
            # print(rnn_v.name, [vocabs[each] for each in n.hidden_v])
        else:
            n.hidden_v, _ = rnn_v.forward(n.parent.iv, y, n.parent.hidden_v)

            if getattr(rnn_h, "conditional_init", False):
                n.left.hidden_h = rnn_h.init_hidden_state(y, n.hidden_v[0])

        n.hidden_h, _ = rnn_h.forward(n.left.iv, y, n.left.hidden_h)

        # print("=>", vocabs[n.iv])
        if has_topo:
            logit, _, _ = fc(n.hidden_h[0], n.hidden_v[0])
        else:
            logit = fc(n.hidden_h[0], n.hidden_v[0])
        n.score = nn.functional.log_softmax(logit, dim=-1)[0, n.iv]

        children = n.children[1:-1] if has_topo else n.children[1:]
        for each in children:
            TreeNode._update_score(each, fc=fc, rnn_h=rnn_h, rnn_v=rnn_v, y=y, has_topo=has_topo)

    def update_score(self, fc, rnn_h: Rnn, rnn_v: Rnn, y, has_topo: bool = False):
        self.hidden_v = rnn_v.init_hidden_state(y)

        if getattr(rnn_h, "conditional_init", False):
            pass
        else:
            hidden_h = rnn_h.init_hidden_state(y)
            self.preorder_walk_children(partial(TreeNode._assign_init_hidden_state, hidden_h=hidden_h))

        children = self.children[1:-1] if has_topo else self.children[1:]
        for each in children:
            TreeNode._update_score(each, fc=fc, rnn_h=rnn_h, rnn_v=rnn_v, y=y, has_topo=has_topo)

    @staticmethod
    def _cumulate_score(n: "TreeNode"):
        sum_children_score = 0
        sum_children_count = 0
        for each in n.children:
            sum_children_score += each.score
            sum_children_count += each.count
        n.score += sum_children_score
        if len(n.children) > 0:
            n.count += sum_children_count - 1

    @staticmethod
    def _adjust_score(n: "TreeNode", alpha: float = 1.0):
        n.score /= (n.count ** alpha)

    def cumulate_score(self, alpha: float = 1.0):
        self.postorder_walk(TreeNode._cumulate_score)
        self.postorder_walk(partial(TreeNode._adjust_score, alpha=alpha))

    @staticmethod
    def _finalize(device, n: "TreeNode"):
        n.children = [TreeNode(device, 3, n)] + n.children + [TreeNode(device, 4, n)]
        n.children[0].right = n.children[1]
        n.children[-1].left = n.children[-2]
        for i in range(1, len(n.children) - 1):
            n.children[i].left = n.children[i - 1]
            n.children[i].right = n.children[i + 1]
            TreeNode._finalize(device, n.children[i])

    @staticmethod
    def build_tree(code, device, add_controls: bool = True):
        root = TreeNode(device, 3)
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
            node = queue[-1].add_child(device, iv)

        if add_controls:
            TreeNode._finalize(device, root)
        return root


class Fusing(nn.Module):

    def __init__(self, decoder_dim, vocab_size, dropout=0.5, enable_topo_predict=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_topo_predict = enable_topo_predict

        self.fc = nn.Linear(decoder_dim * 2, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

        if self.enable_topo_predict:
            self.proj_h = nn.Linear(decoder_dim, 1)
            self.proj_v = nn.Linear(decoder_dim, 1)
            self.offset_h = nn.Parameter(torch.zeros((1, ), dtype=torch.float), requires_grad=True)
            self.offset_v = nn.Parameter(torch.zeros((1, ), dtype=torch.float), requires_grad=True)

        self.init_weights()
    
    def init_weights(self):
        layers = [self.fc]
        if getattr(self, "enable_topo_predict", False):
            layers.extend([self.proj_h, self.proj_v])
        for layer in layers:
            layer.bias.data.fill_(0)
            layer.weight.data.uniform_(-0.1, 0.1)

    def forward(self, h, v):
        cat = torch.cat([h, v], dim=1)
        out = self.fc(self.dropout(cat))

        if getattr(self, "enable_topo_predict", False):
            p_h = nn.functional.sigmoid(self.proj_h(h))
            p_v = nn.functional.sigmoid(self.proj_v(v))
            out += p_h * self.offset_h + p_v * self.offset_v
            return out, p_h, p_v
        
        return out


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, max_len, attention_dim, embed_dim, decoder_dim, vocab_size, 
                 encoder_dim=2048, dropout=0.5, pos_embed=None, 
                 reinforce_learning=False,
                 teacher_forcing_decay: float = 1.0,
                 attention_mode: Optional[str] = None,
                 enable_topo_predict=False,
                 enable_conditional_init=False):
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
        self.attention_mode = attention_mode
        self.double_attention = (attention_mode.endswith("x2") if attention_mode is not None else False)
        self.enable_topo_predict = enable_topo_predict
        self.enable_conditional_init = enable_conditional_init
        self.teacher_forcing_decay = teacher_forcing_decay
        self.teacher_forcing_rate = 1.0
        self.reinforce_learning = reinforce_learning

        if self.attention_mode in [None, "x2"]:
            if not self.double_attention:
                self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
            else:
                self.attention_v = Attention(encoder_dim, decoder_dim, attention_dim)
                self.attention_h = Attention(encoder_dim, decoder_dim, attention_dim)
        elif self.attention_mode.startswith("avg"):
            self.attention = AttentionAvg()
        elif self.attention_mode.startswith("csa"):
            self.double_attention = True
            self.attention_v = None
            self.attention_h = None
        else:  # self.attention_mode == "sum":
            self.attention = AttentionSum()

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        if pos_embed == '1':
            self.pos_embedding = PositionalEmbedding(max_len, embed_dim)
        else:
            self.pos_embedding = None

        if not self.double_attention:
            self.rnn_h = Rnn(self.embedding, self.attention, embed_dim, encoder_dim, decoder_dim,
                             conditional_init=self.enable_conditional_init)
            self.rnn_v = Rnn(self.embedding, self.attention, embed_dim, encoder_dim, decoder_dim)
        else:
            self.rnn_h = Rnn(self.embedding, self.attention_h, embed_dim, encoder_dim, decoder_dim,
                             conditional_init=self.enable_conditional_init)
            self.rnn_v = Rnn(self.embedding, self.attention_v, embed_dim, encoder_dim, decoder_dim)

        # self.decode_step_h = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # self.decode_step_v = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # self.init_h_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        # self.init_h_v = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c_v = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell

        self.fc = Fusing(decoder_dim, vocab_size, dropout=dropout, enable_topo_predict=enable_topo_predict)  # linear layer to find scores over vocabulary
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
        # encoder_dim = encoder_out.size(-1)

        # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        # Embedding
        # embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # if self.pos_embedding is not None:
        #     embeddings = self.pos_embedding(embeddings)

        predict = []
        targets = []
        if return_alphas:
            alphas = []
        if self.enable_topo_predict:
            topos = []

        for bi in range(batch_size):
            # print(bi)

            alphas_nb_h, alphas_nb_v = None, None
            if return_alphas:
                alphas_nb_h = []
                if self.double_attention:
                    alphas_nb_v = alphas_nb_h
                else:
                    alphas_nb_v = []

            tree = TreeNode.build_tree(encoded_captions[bi, :caption_lengths[bi]], device=encoder_out.device)

            # tree.preorder_walk(lambda n: print(n.iv.device))

            prd, tgt = [], []
            topo = None
            if self.enable_topo_predict:
                topo = []
            
            tree.train(self.fc, self.rnn_h, self.rnn_v, encoder_out[bi][None], 
                       prds=prd, tgts=tgt, topos=topo, 
                       alphas_h=alphas_nb_h, alphas_v=alphas_nb_v,
                       teacher_forcing_rate=self.teacher_forcing_rate)

            if topo is not None:          
                topos.extend(topo)
            predict.extend(prd)
            targets.extend(tgt)

            if return_alphas:
                alphas.append(torch.stack(alphas_nb_h).sum(dim=0).squeeze())
                if self.double_attention:
                    alphas.append(torch.stack(alphas_nb_v).sum(dim=0).squeeze())

        r = {
            "predict": torch.cat(predict, dim=0),  # torch.stack(predict),
            "targets": torch.stack(targets),
        }
        if return_alphas:
            r["alphas"] = torch.stack(alphas)
        if self.enable_topo_predict:
            r["topos"] = torch.stack(topos)
        return r
    
    def reinforce_train(self, target: "TreeNode", encoder_out, max_v_len, max_h_len, max_len):
        device = encoder_out.device

        self.eval()
        with torch.no_grad():
            base = self.predict(encoder_out, max_v_len=max_v_len, max_h_len=max_h_len, max_len=max_len, quiet=True)
            # base.cumulate_score()
            # print(base.count, count)
            ted_base = TreeNode.ted(base, target)
        self.train()

        log_probs, rewards = [], []

        root = TreeNode(device, 3)
        root.hidden_v = self.rnn_v.init_hidden_state(encoder_out)
        count = 1

        if not getattr(self.rnn_h, "conditional_init", False):
            init_hidden_h = self.rnn_h.init_hidden_state(encoder_out)

        queue = [root]
        while len(queue) != 0:
            parent_node = queue.pop()
            if parent_node.height >= max_v_len - 2 or count >= max_len:
                break

            left_node = parent_node.add_child(device, 3)
            count += 1

            if count >= max_len:
                break

            hidden_v, _ = self.rnn_v.forward(parent_node.iv, encoder_out, parent_node.hidden_v)

            if getattr(self.rnn_h, "conditional_init", False):
                left_node.hidden_h = self.rnn_h.init_hidden_state(encoder_out, hidden_v[0])
            else:
                left_node.hidden_h = init_hidden_h

            for i in range(max_h_len - 1):
                hidden_h, _ = self.rnn_h.forward(left_node.iv, encoder_out, left_node.hidden_h)
                logit = self.fc(hidden_h[0], hidden_v[0])
                probs = torch.softmax(logit, dim=-1)[0]
                # print(probs.shape)
                iv = torch.multinomial(probs, 1).item()
                # print(iv.shape)
                log_probs.append(torch.log(probs[iv]))
                # if iv == 0:
                #     iv = 4
                if i == max_h_len - 2:
                    iv = 4
                # print(ted_before)
                node = parent_node.add_child(device, iv)
                node.score = logit
                count += 1
                # node.score = nn.functional.log_softmax(logit, dim=-1)[0, iv]
                ted = TreeNode.ted(root, target)
                # print(-(ted - ted_base))
                rewards.append(-(ted - ted_base))
                if iv == 4 or count >= max_len:
                    break
                node.hidden_v = hidden_v
                node.hidden_h = hidden_h
                queue.append(node)
                left_node = node

        # root.cumulate_score()
        return root, log_probs, rewards
    
    def predict(self, encoder_out, max_v_len, max_h_len, 
                max_len = None, cumulate_score = False,
                verbose=False, quiet: bool = False):
        if verbose:
            from tqdm.notebook import tqdm

        device = encoder_out.device

        root = TreeNode(device, 3)
        root.hidden_v = self.rnn_v.init_hidden_state(encoder_out)
        count = 1

        if not getattr(self.rnn_h, "conditional_init", False):
            init_hidden_h = self.rnn_h.init_hidden_state(encoder_out)

        queue = [root]
        while len(queue) != 0:
            parent_node = queue.pop()
            if parent_node.height >= max_v_len - 2:
                if not quiet:
                    print("max height exceeded!")
                break
            if count >= max_len:
                if not quiet:
                    print("max length exceeded!")
                break

            left_node = parent_node.add_child(device, 3)
            count += 1

            if count >= max_len:
                if not quiet:
                    print("max length exceeded!")
                break

            hidden_v, _ = self.rnn_v.forward(parent_node.iv, encoder_out, parent_node.hidden_v)

            if getattr(self.rnn_h, "conditional_init", False):
                left_node.hidden_h = self.rnn_h.init_hidden_state(encoder_out, hidden_v[0])
            else:
                left_node.hidden_h = init_hidden_h

            tbar = range(max_h_len - 1)
            if verbose:
                tbar = tqdm(tbar)
            for i in tbar:
                if verbose:
                    tbar.set_description(f"@{parent_node.id}")
                hidden_h, _ = self.rnn_h.forward(left_node.iv, encoder_out, left_node.hidden_h)
                logit = self.fc(hidden_h[0], hidden_v[0])
                iv = torch.argmax(nn.functional.softmax(logit, dim=-1), dim=-1)[0]
                if iv == 0:
                    iv = 4
                if i == max_h_len - 2:
                    iv = 4
                    if not quiet:
                        print("max width exceeded!")
                node = parent_node.add_child(device, iv)
                node.score = nn.functional.log_softmax(logit, dim=-1)[0, iv]
                count += 1
                if iv == 4:
                    break
                if count >= max_len:
                    if not quiet:
                        print("max length exceeded!")
                    break
                node.hidden_v = hidden_v
                node.hidden_h = hidden_h
                queue.append(node)
                left_node = node

                if verbose:
                    print(f"{i:3}", f"{node.iv}", f"{len(parent_node.children)}", root.build_list())
            # else:
                # print("max width exceed!")

        if cumulate_score:
            root.cumulate_score()
        return root
    
    def predict_score(self, encoder_out, tree: "TreeNode", verbose=False):
        enable_topo_predict = getattr(self, "enable_topo_predict", False)
        tree.update_score(self.fc, self.rnn_h, self.rnn_v, encoder_out, enable_topo_predict)
        tree.cumulate_score()


class ImageCaptionWithBit(nn.Module):
    """
    Image captioning with bi-directional tree rnn.
    """

    def __init__(self, 
                 resnet, 
                 vocab_size: int, 
                 max_len,              
                 reinforce_learning = None,
                 teacher_forcing_decay = None,
                 attention_mode = None,
                 enable_topo_predict = None,
                 enable_attention_regularization = None,
                 enable_conditional_init = None,
                 proof_of_concept: bool = False):
        super().__init__()
        self.proof_of_concept: bool = proof_of_concept
        self.attention_mode = attention_mode
        self.enable_topo_predict = (enable_topo_predict == '1')
        self.enable_attention_regularization = (enable_attention_regularization == '1')
        self.enable_conditional_init = (enable_conditional_init == '1')
        self.teacher_forcing_decay = (float(teacher_forcing_decay) if teacher_forcing_decay is not None else 1.0)
        self.reinforce_learning = (reinforce_learning == '1')

        print("[params] {}".format(", ".join([
            f"{k}={v}" for k, v in {
                "proof_of_concept": self.proof_of_concept,
                "attention_mode": self.attention_mode,
                "enable_topo_predict": self.enable_topo_predict,
                "enable_attention_regularization": self.enable_attention_regularization,
                "enable_conditional_init": self.enable_conditional_init,
                "teacher_forcing_decay": self.teacher_forcing_decay,
                "reinforce_learning": self.reinforce_learning
            }.items()
        ])))

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.alpha_c = 1.
        self.encoder = Encoder(resnet)
        self.decoder = DecoderWithAttention(max_len,
                                            attention_dim=512,
                                            embed_dim=512,
                                            decoder_dim=512,
                                            vocab_size=vocab_size,
                                            dropout=0.2,
                                            reinforce_learning=self.reinforce_learning,
                                            teacher_forcing_decay=self.teacher_forcing_decay,
                                            attention_mode=self.attention_mode,
                                            enable_topo_predict=self.enable_topo_predict,
                                            enable_conditional_init=self.enable_conditional_init)
        self.criterion = nn.CrossEntropyLoss()
        
    def begin_epoch(self, epoch):
        if epoch < 10 or self.teacher_forcing_decay == 1.0:
            return
        self.decoder.teacher_forcing_rate *= self.decoder.teacher_forcing_decay
        print(f"teacher_forcing_rate = {self.decoder.teacher_forcing_rate}")

    def end_batch(self, outputs, batch, optimizer):
        optimizer.zero_grad()

        imgs = batch["image"]
        caps = batch["code"].long()
        batch_size = imgs.size(0)

        imgs = self.encoder(imgs)

        log_probs, rewards = [], []
        for i in range(batch_size):
            target = TreeNode.build_tree(caps[i], device="cpu")
            _, log_prob, reward = self.decoder.reinforce_train(target, imgs[i][None], 
                                                               max_v_len=15, max_h_len=10,
                                                               max_len=self.max_len)
            log_probs.extend(log_prob)
            rewards.extend(reward)

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + 0.99 * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R) 

        loss = torch.stack(loss).sum()
        loss.backward()

        optimizer.step()

        outputs["loss/rf"] = loss.item()
        return outputs

    def forward(self, batch):
        imgs = batch["image"]
        caps = batch["code"].long()
        caplens = batch["code_len"]

        # Forward prop.
        imgs = self.encoder(imgs)
        # batch_size = imgs.size(0)

        outs = self.decoder(imgs, caps, caplens, return_alphas=self.enable_attention_regularization)
        scores = outs["predict"]
        targets = outs["targets"]

        loss = self.criterion(scores, targets)

        # print(scores.shape, targets.shape)
        # print(alphas.shape)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        # targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # scores = nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        # targets = nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss

        r = {
            "loss": loss, 
            "scores": torch.nn.functional.softmax(scores, dim=-1), 
            # "sources": torch.argmax(torch.nn.functional.softmax(scores, dim=-1), dim=-1),
            "targets": targets
        }

        if self.enable_topo_predict:
            topos = outs["topos"]
            p_h, t_h, p_v, t_v = topos.chunk(4, dim=-1)
            p = torch.cat([p_h, p_v], dim=0).squeeze(0)
            t = torch.cat([t_h, t_v], dim=0).squeeze(0)
            loss_topo = nn.functional.binary_cross_entropy_with_logits(p, t)
            # print(loss_topo)
            loss += loss_topo
            r["loss/topo"] = loss_topo

        if self.enable_attention_regularization:
            alphas = outs["alphas"]
            # Add doubly stochastic attention regularization
            loss_reg = self.alpha_c * ((1. - alphas) ** 2).mean()
            loss += loss_reg
            r["loss/reg"] = loss_reg

        return r
    
    # def predict(self, batch, hiddens = None):
    #     imgs = batch["image"]
    #     caps = batch["code"].long()

    #     imgs = self.encoder(imgs)

    #     preds, _, hiddens = self.decoder.predict(imgs, caps, hiddens)

    #     return {
    #         "scores": preds,
    #         "hiddens": hiddens
    #     }

    def predict(self, images, max_v_len, max_h_len, verbose=False, quiet=False):
        batch_size = images.size(0)
        
        # with torch.no_grad() should be called outside this scope.
        encoder_out = self.encoder(images)
        # encoder_dim = encoder_out.size(-1)
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        trees = []

        for i in range(batch_size):
            if verbose:
                print(f"Sample {i} =>")
            tree = self.decoder.predict(encoder_out[i][None], 
                                        max_v_len, max_h_len,
                                        max_len=self.max_len, 
                                        verbose=verbose, quiet=quiet)  # (batch_size, decoder_dim)
            trees.append(tree)

        return trees
    
    def predict_score(self, images, trees, verbose=False):
        batch_size = images.size(0)
        assert len(trees) == batch_size
        
        # with torch.no_grad() should be called outside this scope.
        encoder_out = self.encoder(images)
        # encoder_dim = encoder_out.size(-1)
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        
        for i, tree in enumerate(trees):
            self.decoder.predict_score(encoder_out[i][None], tree, verbose=verbose)
