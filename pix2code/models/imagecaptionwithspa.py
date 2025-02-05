import torch
import torch.nn as nn
import torchvision
import math
import copy
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self):
        super().__init__()
        # Build heads.
        num_classes = 90
        d_model = 256
        dim_feedforward = 2048
        nhead = 8
        dropout = 0.0
        activation = 'relu'
        num_heads = 6
        rcnn_head = RCNNHead(d_model, num_classes, dim_feedforward, nhead, dropout, activation)        
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = False
        
        # Init parameters.
        self.use_focal = True
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = 0.01
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    def forward(self, features, init_bboxes, init_features):

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        
        # print(init_features.shape)
        init_features = init_features[None].repeat(1, bs, 1)  # em...
        # print(init_features.shape)
        proposal_features = init_features.clone()

        # print(proposal_features.shape)
        
        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features)

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.box_pooler = ROIPooler(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )

        self.d_model = d_model

        approx_scale = float(14) / float(256)
        self.scale = 2 ** float(torch.tensor(approx_scale).log2().round())

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = 1
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = 3
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = True
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        # print("bboxes", bboxes.shape)
        # print("pro_features", pro_features.shape)

        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = []
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = self.box_pooler(features, proposal_boxes)   
        # print("roi_features", roi_features.shape)  # (200, 256, 7, 7)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)        

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features
    

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_dim = 256
        self.dim_dynamic = 64
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = 7
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        # print("pro_features", pro_features.shape)
        # print("roi_features", roi_features.shape)

        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, resnet):
        super(Encoder, self).__init__()
        # Extract 4 main layers
        self.body = create_feature_extractor(
            resnet, return_nodes={f'layer{k}': str(v) for v, k in enumerate([1, 2, 3, 4])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 4, 256, 256)
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

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        batch_size, encoder_dim, w, h = encoder_out.shape

        ev = encoder_out.permute(0, 2, 3, 1).view(batch_size, -1, encoder_dim)

        att1 = self.encoder_att(ev)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (ev * alpha.unsqueeze(2))  # (batch_size, num_pixels, encoder_dim)
        # .sum(dim=1)
        aw = attention_weighted_encoding.view(batch_size, w, h, encoder_dim).permute(0, 3, 1, 2)

        return attention_weighted_encoding.sum(dim=1), aw, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, visnet, attention_dim, embed_dim, decoder_dim, vocab_size, 
                 encoder_dim=256, dropout=0.5, embed_parent=None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        if visnet is not None:
            visnet.eval()
        self.visnet = visnet
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embed_parent = embed_parent

        #
        self.num_proposals = 20
        self.hidden_dim = 256

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.head = DynamicHead()
        
        #
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        if self.embed_parent == 1:
            self.decode_step = nn.LSTMCell(2 * embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        else:
            self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc_cls = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.fc_box = nn.Linear(196, 4)  #
        # self.fc_equ = nn.Linear(decoder_dim, 1)  #
        # self.fc_ign = nn.Linear(decoder_dim, 1)  #
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc_cls.bias.data.fill_(0)
        self.fc_cls.weight.data.uniform_(-0.1, 0.1)
        self.fc_box.bias.data.fill_(0)
        self.fc_box.weight.data.uniform_(-0.1, 0.1)
        # self.fc_equ.bias.data.fill_(0)
        # self.fc_equ.weight.data.uniform_(-0.1, 0.1)
        # self.fc_ign.bias.data.fill_(0)
        # self.fc_ign.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, eo1, eo2, eo3, eo4):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        bs, ed, _, _ = eo1.shape
        
        mean_encoder_out = eo1.view(bs, ed, -1).mean(dim=-1)
        mean_encoder_out += eo2.view(bs, ed, -1).mean(dim=-1) 
        mean_encoder_out += eo3.view(bs, ed, -1).mean(dim=-1) 
        mean_encoder_out += eo4.view(bs, ed, -1).mean(dim=-1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, images, encoder_out, encoded_captions, caption_lengths, pivs):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        # batch_size = encoder_out.size(0)
        # encoder_dim = encoder_out.size(-1)
        # vocab_size = self.vocab_size


        # # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        images = images[sort_ind]

        #
        eo1, eo2, eo3, eo4 = encoder_out
        eo1 = eo1[sort_ind]
        eo2 = eo2[sort_ind]
        eo3 = eo3[sort_ind]
        eo4 = eo4[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        if self.embed_parent is not None:
            pivs_sorted = pivs[sort_ind]
            embeddings_parent = self.embedding(pivs_sorted)

        # Initialize LSTM state
        h, c = self.init_hidden_state(eo1, eo2, eo3, eo4)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        # preds_cls = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        # preds_box = torch.zeros(batch_size, max(decode_lengths), 4).to(encoder_out.device)
        # # preds_equ = torch.zeros(batch_size, max(decode_lengths), 1).to(encoder_out.device)
        # # preds_ign = torch.zeros(batch_size, max(decode_lengths), 1).to(encoder_out.device)
        # alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            feats = []
            for each in [eo1, eo2, eo3, eo4]:
                attention_weighted_encoding, aw, alpha = self.attention(each[:batch_size_t], h[:batch_size_t])
                feats.append(aw)

            # Prepare Proposals.
            proposal_boxes = self.init_proposal_boxes.weight.clone()
            proposal_boxes = torchvision.ops.box_convert(proposal_boxes, "cxcywh", "xyxy")
            proposal_boxes = proposal_boxes[None] * 256
            proposal_boxes = proposal_boxes.repeat(batch_size_t, 1, 1)

            outputs_class, outputs_coord = self.head(feats, proposal_boxes, self.init_proposal_features.weight)
            print(outputs_class.shape, outputs_coord.shape)
            # print(outputs_coord[0])
            outputs_class, outputs_coord = outputs_class[-1], outputs_coord[-1]

            print(outputs_class)
            print(outputs_coord)

            # outputs_scores = torch.zeros((batch_size_t, 100), dtype=torch.float32).to(images.device)

            # with torch.no_grad():
            #     for b in range(batch_size_t):
            #         masks = torch.zeros((20, 1, 256, 256), dtype=torch.float32).to(images.device)
            #         for j in range(1, 101):
            #             x1, y1, x2, y2 = outputs_coord[b, j - 1].int().unbind(dim=-1)
            #             masks[j % 20, 0, x1:y1, x2:y2] = 1
            #             if j % 20 == 0:
            #                 imgs = torch.cat([
            #                     images[b, :3][None].repeat(20, 1, 1, 1),
            #                     masks
            #                 ], dim=1)
            #                 scores = nn.functional.sigmoid(self.visnet(imgs)).flatten()
            #                 outputs_scores[b, j - 20:j] = scores
            #                 masks[:, :, :, :] = 0
            # # assert False

            # topks = outputs_scores.topk(10, dim=1)
            # print(outputs_class[topks.indices])
            # print(outputs_coord[topks.indices])


            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            
            # print("gate", attention_weighted_encoding.shape, gate.shape)
            
            attention_weighted_encoding = gate * attention_weighted_encoding
            if self.embed_parent == "cat" or self.embed_parent == 1:
                h, c = self.decode_step(
                    torch.cat([embeddings_parent[:batch_size_t, :], embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            elif self.embed_parent == "add" or self.embed_parent == 2:
                new_embeddings = embeddings[:, t, :] + embeddings_parent
                h, c = self.decode_step(
                    torch.cat([new_embeddings[:batch_size_t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            # h = self.dropout(h)
            # preds_cls[:batch_size_t, t, :] = self.fc_cls(h)  # (batch_size_t, vocab_size)
            # preds_box[:batch_size_t, t, :] = self.fc_box(alpha).sigmoid()
            # # preds_equ[:batch_size_t, t, :] = self.fc_equ(h)
            # # preds_ign[:batch_size_t, t, :] = self.fc_ign(h)
            # alphas[:batch_size_t, t, :] = alpha

            assert False

        # return preds_cls, preds_box, preds_equ, preds_ign, encoded_captions, decode_lengths, alphas, sort_ind
        return preds_cls, preds_box, encoded_captions, decode_lengths, alphas, sort_ind
    
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        if self.embed_parent is not None:
            embeddings_parent = self.embedding(pivs)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        for i in range(batch_size):
            for t in range(decode_lengths[i]):

                attention_weighted_encoding, alpha = self.attention(encoder_out[i][None], h[i][None])
        
                # Prepare Proposals.
                proposal_boxes = self.init_proposal_boxes.weight.clone()
                proposal_boxes = torchvision.ops.box_convert(proposal_boxes, "cxcywh", "xyxy")
                proposal_boxes = proposal_boxes[None] * 256

                feats = encoder_out[i][None] * alpha.unsqueeze(2) 
                feats = feats.view(1, 14, 14, 2048)
                feats = feats.permute(0, 3, 1, 2) 

                outs = self.head(feats, proposal_boxes, self.init_proposal_features.weight)
                print(outs)

        assert False
        # Create tensors to hold word predicion scores and alphas
        preds_cls = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        preds_box = torch.zeros(batch_size, max(decode_lengths), 4).to(encoder_out.device)
        # preds_equ = torch.zeros(batch_size, max(decode_lengths), 1).to(encoder_out.device)
        # preds_ign = torch.zeros(batch_size, max(decode_lengths), 1).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            # Prepare Proposals.
            proposal_boxes = self.init_proposal_boxes.weight.clone()
            proposal_boxes = torchvision.ops.box_convert(proposal_boxes, "cxcywh", "xyxy")
            proposal_boxes = proposal_boxes[None] * 256
            proposal_boxes = proposal_boxes.repeat(batch_size_t, 1, 1)

            feats = encoder_out[:batch_size_t] * alpha.unsqueeze(2) 
            feats = feats.view(batch_size_t, 14, 14, 2048)
            feats = feats.permute(0, 3, 1, 2) 

            outs = self.head(feats, proposal_boxes, self.init_proposal_features.weight)
            print(outs)

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            if self.embed_parent == "cat" or self.embed_parent == 1:
                h, c = self.decode_step(
                    torch.cat([embeddings_parent[:batch_size_t, :], embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            elif self.embed_parent == "add" or self.embed_parent == 2:
                new_embeddings = embeddings[:, t, :] + embeddings_parent
                h, c = self.decode_step(
                    torch.cat([new_embeddings[:batch_size_t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            h = self.dropout(h)
            preds_cls[:batch_size_t, t, :] = self.fc_cls(h)  # (batch_size_t, vocab_size)
            preds_box[:batch_size_t, t, :] = self.fc_box(alpha).sigmoid()
            # preds_equ[:batch_size_t, t, :] = self.fc_equ(h)
            # preds_ign[:batch_size_t, t, :] = self.fc_ign(h)
            alphas[:batch_size_t, t, :] = alpha

        # return preds_cls, preds_box, preds_equ, preds_ign, encoded_captions, decode_lengths, alphas, sort_ind
        return preds_cls, preds_box, encoded_captions, decode_lengths, alphas, sort_ind


class ImageCaptionWithSpa(nn.Module):

    def __init__(self, resnet, visnet, vocab_size: int, embed_parent: str = None):
        super().__init__()
        self.alpha_c = 1.
        self.encoder = Encoder(resnet)
        self.decoder = DecoderWithAttention(visnet, 
                                            attention_dim=512,
                                            embed_dim=512,
                                            decoder_dim=512,
                                            vocab_size=vocab_size,
                                            dropout=0.5,
                                            embed_parent=embed_parent)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_box = torchvision.ops.generalized_box_iou_loss
        # self.criterion_equ = nn.BCEWithLogitsLoss()
        # self.criterion_ign = nn.BCEWithLogitsLoss()
        self.log_vars = nn.Parameter(torch.zeros((2, ), requires_grad=True))
        
    def forward(self, batch):
        imgs = batch["image"]
        caps = batch["code"].long()
        caplens = batch["code_len"]
        pivs = batch["piv"].long()
        
        boxs = batch["rect"]
        # equs = batch["equal"].float()
        # igns = batch["ignore"].float()

        # Forward prop.
        encoded_imgs = self.encoder(imgs)
        # preds_cls, preds_box, preds_equ, preds_ign, caps_sorted, decode_lengths, alphas, sort_ind = \
        #     self.decoder(imgs, caps, caplens)
        preds_cls, preds_box, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, encoded_imgs, caps, caplens, pivs)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        truth_cls = caps_sorted[:, 1:]
        truth_box = boxs[sort_ind, 1:]
        # truth_equ = equs[sort_ind, 1:]
        # truth_ign = igns[sort_ind, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        preds_cls = nn.utils.rnn.pack_padded_sequence(preds_cls, decode_lengths, batch_first=True).data
        truth_cls = nn.utils.rnn.pack_padded_sequence(truth_cls, decode_lengths, batch_first=True).data

        # Calculate loss
        loss_cls = self.criterion_cls(preds_cls, truth_cls)

        # Add doubly stochastic attention regularization
        loss_cls += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        #
        # preds_equ = nn.utils.rnn.pack_padded_sequence(preds_equ, decode_lengths, batch_first=True).data
        # truth_equ = nn.utils.rnn.pack_padded_sequence(truth_equ, decode_lengths, batch_first=True).data

        # loss_equ = self.criterion_equ(preds_equ.squeeze(), truth_equ)

        #
        # preds_ign = nn.utils.rnn.pack_padded_sequence(preds_ign, decode_lengths, batch_first=True).data
        # truth_ign = nn.utils.rnn.pack_padded_sequence(truth_ign, decode_lengths, batch_first=True).data

        # loss_ign = self.criterion_ign(preds_ign.squeeze(), truth_ign)

        #
        preds_box = nn.utils.rnn.pack_padded_sequence(preds_box, decode_lengths, batch_first=True).data
        truth_box = nn.utils.rnn.pack_padded_sequence(truth_box, decode_lengths, batch_first=True).data

        # box_masks = (1 - truth_equ.long()) * (1 - truth_ign.long())
        box_masks = (truth_cls > 7)

        preds_box = preds_box[box_masks]
        truth_box = truth_box[box_masks]

        preds_box = torchvision.ops.box_convert(preds_box, "cxcywh", "xyxy")
        truth_box = torchvision.ops.box_convert(truth_box, "cxcywh", "xyxy")

        loss_box = self.criterion_box(preds_box, truth_box, reduction="mean")

        #
        # loss = loss_cls + loss_equ + loss_ign + loss_box
        p1 = 0.5 * torch.exp(-self.log_vars[0])
        p2 = 0.5 * torch.exp(-self.log_vars[1])
        loss = p1 * loss_cls + p2 * loss_box + self.log_vars[0] + self.log_vars[1]

        return {
            "loss": loss, 
            "loss/cls": loss_cls,
            # "loss/equ": loss_equ,
            # "loss/ign": loss_ign,
            "loss/box": loss_box,
            "scores": preds_cls,  
            "targets": truth_cls,
            "preds_box": preds_box,
            "truth_box": truth_box
        }