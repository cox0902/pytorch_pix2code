from typing import *
import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torchvision
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms.v2


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    dice_score = soft_tversky_score(output, target, 0.5, 0.5, smooth, eps, dims)
    return dice_score


def soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    """Tversky loss

    References:
        https://arxiv.org/pdf/2302.05666
        https://arxiv.org/pdf/2303.16296

    """
    assert output.size() == target.size()

    if dims is not None:
        output_sum = torch.sum(output, dim=dims)
        target_sum = torch.sum(target, dim=dims)
        difference = LA.vector_norm(output - target, ord=1, dim=dims)
    else:
        output_sum = torch.sum(output)
        target_sum = torch.sum(target)
        difference = LA.vector_norm(output - target, ord=1)

    intersection = (output_sum + target_sum - difference) / 2  # TP
    fp = output_sum - intersection
    fn = target_sum - intersection

    tversky_score = (intersection + smooth) / (
        intersection + alpha * fp + beta * fn + smooth
    ).clamp_min(eps)
    return tversky_score


class DiceLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceLoss, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        return self.conv(x1)

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UpNet(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # print(in_channels)
        # print(skip_channels)
        # print(out_channels)

        self.blocks = nn.ModuleList()
        for in_c, skip_c, out_c in zip(in_channels, skip_channels, out_channels):
            block = Up(in_c, skip_c, out_c, False)
            self.blocks.append(block)

        self.out = OutConv(out_c, 1)

    def forward(self, *features):
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        # spatial_shapes = [feature.shape[2:] for feature in features]
        # spatial_shapes = spatial_shapes[::-1]
        
        # features = features[1:]
        features = features[::-1]

        x = features[0]
        skip_connections = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            # height, width = spatial_shapes[i + 1]
            # print(x.shape, height, width)
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, skip_connection)
        
        # print(x.shape)
        return self.out(x)


class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, resnet):
        super(Encoder, self).__init__()
        # Extract 4 main layers
        return_nodes = { f'relu': '0' }
        for k in [1, 2, 3, 4]:
            return_nodes[f'layer{k}'] = str(k)
        self.body = create_feature_extractor(resnet, return_nodes=return_nodes)

    def forward(self, images):
        x = self.body(images)
        return [x[str(k)] for k in range(5)]

    def freeze(self):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.body.parameters():
            p.requires_grad = False
        

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
        attention_weighted_encoding = (ev * alpha.unsqueeze(2))  # (batch_size, encoder_dim)

        aw = attention_weighted_encoding.view(batch_size, w, h, encoder_dim).permute(0, 3, 1, 2)

        return attention_weighted_encoding.sum(dim=1), aw, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5,
                 embed_parent=None):
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
        self.embed_parent = embed_parent

        encoder_channels = (4, 64, 256, 512, 1024, 2048)
        decoder_channels = (256, 128, 64, 32, 16)
        self.upnet = UpNet(encoder_channels, decoder_channels)

        self.att1 = Attention(64, decoder_dim, attention_dim)  # attention network
        self.att2 = Attention(256, decoder_dim, attention_dim)  # attention network
        self.att3 = Attention(512, decoder_dim, attention_dim)  # attention network
        self.att4 = Attention(1024, decoder_dim, attention_dim)  # attention network
        self.att5 = Attention(2048, decoder_dim, attention_dim)  # attention network

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
        # self.fc_box = nn.Linear(196, 4)  #
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
        # self.fc_box.bias.data.fill_(0)
        # self.fc_box.weight.data.uniform_(-0.1, 0.1)
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

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        bs, ed, _, _ = encoder_out.shape

        mean_encoder_out = encoder_out.view(bs, ed, -1).mean(dim=-1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, pivs):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoded_captions.size(0)
        # encoder_dim = encoder_out.size(-1)
        # vocab_size = self.vocab_size

        # # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        # encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        #
        eo1, eo2, eo3, eo4, eo5 = encoder_out
        eo1 = eo1[sort_ind]
        eo2 = eo2[sort_ind]
        eo3 = eo3[sort_ind]
        eo4 = eo4[sort_ind]
        eo5 = eo5[sort_ind]
        # print(eo5.shape)

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        if self.embed_parent is not None:
            pivs_sorted = pivs[sort_ind]
            embeddings_parent = self.embedding(pivs_sorted)

        # Initialize LSTM state
        h, c = self.init_hidden_state(eo5)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        preds_cls = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoded_captions.device)
        preds_box = torch.zeros(batch_size, max(decode_lengths), 256, 256).to(encoded_captions.device)
        # preds_equ = torch.zeros(batch_size, max(decode_lengths), 1).to(encoder_out.device)
        # preds_ign = torch.zeros(batch_size, max(decode_lengths), 1).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), 64).to(encoded_captions.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            awe1, aw1, alpha1 = self.att1(eo1[:batch_size_t], h[:batch_size_t])
            awe2, aw2, alpha2 = self.att2(eo2[:batch_size_t], h[:batch_size_t])
            awe3, aw3, alpha3 = self.att3(eo3[:batch_size_t], h[:batch_size_t])
            awe4, aw4, alpha4 = self.att4(eo4[:batch_size_t], h[:batch_size_t])
            awe5, aw5, alpha5 = self.att5(eo5[:batch_size_t], h[:batch_size_t])
            
            out = self.upnet(aw1, aw2, aw3, aw4, aw5)
            preds_box[:batch_size_t, t, :, :] = out[:, 0, :, :]  # .cpu()

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * awe5
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
            preds_cls[:batch_size_t, t, :] = self.fc_cls(h)  # .cpu()  # (batch_size_t, vocab_size)
            # preds_equ[:batch_size_t, t, :] = self.fc_equ(h)
            # preds_ign[:batch_size_t, t, :] = self.fc_ign(h)
            alphas[:batch_size_t, t, :] = alpha5  # .cpu()

        # return preds_cls, preds_box, preds_equ, preds_ign, encoded_captions, decode_lengths, alphas, sort_ind
        return preds_cls, preds_box, encoded_captions, decode_lengths, alphas, sort_ind


class ImageCaptionWithMsk(nn.Module):

    def __init__(self, resnet, vocab_size: int, embed_parent: str = None, mix: str = None, freeze: bool = False,
                 resize: str = None):
        super().__init__()
        self.mix = mix if mix is not None else "all"
        self.alpha_c = 1.
        self.encoder = Encoder(resnet)
        if freeze:
            self.encoder.freeze()
        self.decoder = DecoderWithAttention(attention_dim=512,
                                            embed_dim=512,
                                            decoder_dim=512,
                                            vocab_size=vocab_size,
                                            dropout=0.2,
                                            embed_parent=embed_parent)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_dice = DiceLoss()
        self.criterion_bcls = nn.BCEWithLogitsLoss()
        # self.criterion_ign = nn.BCEWithLogitsLoss()
        if self.mix == "all":
            self.log_vars = nn.Parameter(torch.zeros((2, ), requires_grad=True))
        self.resize = resize if resize is None else int(resize)
        
    def forward(self, batch):
        imgs = batch["image"]
        caps = batch["code"].long()
        caplens = batch["code_len"]
        pivs = batch["piv"].long()
        
        masks = batch["mask"]
        # equs = batch["equal"].float()
        # igns = batch["ignore"].float()

        # Forward prop.
        encoded_imgs = self.encoder(imgs)
        # preds_cls, preds_box, preds_equ, preds_ign, caps_sorted, decode_lengths, alphas, sort_ind = \
        #     self.decoder(imgs, caps, caplens)
        preds_cls, preds_box, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            encoded_imgs, caps, caplens, pivs)

        if self.mix in ["all", "cls"] or not self.training:
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            truth_cls = caps_sorted[:, 1:]  # .cpu()
            # truth_box = masks[sort_ind.cpu(), 1:]
            # truth_equ = equs[sort_ind, 1:]
            # truth_ign = igns[sort_ind, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            preds_cls = nn.utils.rnn.pack_padded_sequence(preds_cls, decode_lengths, batch_first=True).data
            truth_cls = nn.utils.rnn.pack_padded_sequence(truth_cls, decode_lengths, batch_first=True).data

            if self.training:
                # Calculate loss
                loss_cls = self.criterion_cls(preds_cls, truth_cls)

        # Add doubly stochastic attention regularization
        # loss_cls += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        #
        # preds_equ = nn.utils.rnn.pack_padded_sequence(preds_equ, decode_lengths, batch_first=True).data
        # truth_equ = nn.utils.rnn.pack_padded_sequence(truth_equ, decode_lengths, batch_first=True).data

        # loss_equ = self.criterion_equ(preds_equ.squeeze(), truth_equ)

        #
        # preds_ign = nn.utils.rnn.pack_padded_sequence(preds_ign, decode_lengths, batch_first=True).data
        # truth_ign = nn.utils.rnn.pack_padded_sequence(truth_ign, decode_lengths, batch_first=True).data

        # loss_ign = self.criterion_ign(preds_ign.squeeze(), truth_ign)

        if self.mix in ["all", "box"] or not self.training:
            truth_box = masks[sort_ind, 1:]

            #
            preds_box = nn.utils.rnn.pack_padded_sequence(preds_box, decode_lengths, batch_first=True).data
            truth_box = nn.utils.rnn.pack_padded_sequence(truth_box, decode_lengths, batch_first=True).data

            if self.resize is not None:
                preds_box = F.interpolate(preds_box, size=self.resize, mode="nearest")
                truth_box = F.interpolate(truth_box, size=self.resize, mode="nearest")

            # print(preds_box.shape)
            # print(truth_box.shape)
            if self.training:
                loss_dice = self.criterion_dice(preds_box, truth_box)
                loss_bcls = self.criterion_bcls(preds_box, truth_box)
                loss_box = 0.8 * loss_bcls + 0.2 * loss_dice

        # box_masks = (1 - truth_equ.long()) * (1 - truth_ign.long())
        # box_masks = (truth_cls > 7)

        # preds_box = preds_box[box_masks]
        # truth_box = truth_box[box_masks]

        # preds_box = torchvision.ops.box_convert(preds_box, "cxcywh", "xyxy")
        # truth_box = torchvision.ops.box_convert(truth_box, "cxcywh", "xyxy")

        # loss_box = self.criterion_box(preds_box, truth_box, reduction="mean")

        #
        # loss = loss_cls + loss_equ + loss_ign + loss_box
        return_dict = {}

        if self.training:

            if self.mix == "all":
                p1 = 0.5 * torch.exp(-self.log_vars[0])
                p2 = 0.5 * torch.exp(-self.log_vars[1])
                loss = p1 * loss_cls + p2 * loss_box + self.log_vars[0] + self.log_vars[1]

                return_dict["loss"] = loss
                return_dict["loss/cls"] = loss_cls
                return_dict["loss/box"] = loss_box
            elif self.mix == "cls":
                return_dict["loss"] = loss_cls
            elif self.mix == "box":
                return_dict["loss"] = loss_box
                
            return return_dict
        else:
            return_dict["scores"] = preds_cls  
            return_dict["targets"] = truth_cls
            return_dict["preds_box"] = preds_box
            return_dict["truth_box"] = truth_box
            return return_dict