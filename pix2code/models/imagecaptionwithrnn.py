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

        return attention_weighted_encoding, alpha


class TokenDecoder(nn.Module):

    def __init__(self, embed_dim, encoder_dim, decoder_dim, attention_dim, vocab_size, 
                 dropout=0.2, proof_of_concept=False, 
                 enable_attention=False,
                 enable_encoder=None,
                 tf_token_decoder=1.0,
                 disable_cat=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.proof_of_concept = proof_of_concept
        self.enable_attention = enable_attention
        self.tf_token_decoder = tf_token_decoder
        self.disable_cat = disable_cat

        if self.enable_attention:
            self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
            self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
            self.sigmoid = nn.Sigmoid()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)

        if enable_encoder is not None:
            self.encode_step = nn.LSTMCell(encoder_dim, encoder_dim, bias=True)
        else:
            self.encode_step = None

        if self.disable_cat:
            self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        else:
            self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c
    
    def forward(self, encoder_out, targets, target_lengths):
        
        target_lengths, sort_ind = target_lengths.sort(dim=0, descending=True)
        targets = targets[sort_ind]  # (batch_size, seq_len)
        target_lengths = target_lengths[sort_ind]
        encoder_out = encoder_out[sort_ind]

        if self.encode_step is not None:
            hiddens = None
            for t in range(encoder_out.size(1)):
                hiddens = self.encode_step(encoder_out[:, t, :], hiddens)
            encoder_out = hiddens[0]

        h, c = self.init_hidden_state(encoder_out)

        embeddings = self.embedding(targets)

        decode_lengths = (target_lengths - 1).tolist()

        predictions = torch.zeros(len(decode_lengths), max(decode_lengths), self.vocab_size).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            if self.enable_attention:
                attention_weighted_encoding, _ = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
            else:
                attention_weighted_encoding = encoder_out[:batch_size_t]

            ta_embeddings = embeddings[:batch_size_t, t, :]
            if self.training and t != 0:
                tf_preds = torch.argmax(predictions[:batch_size_t, t - 1, :], dim=-1)
                tf_embeddings = self.embedding(tf_preds)
                tf_mask = (torch.rand(batch_size_t) < self.tf_token_decoder)
                tf_embeddings[tf_mask, :] = ta_embeddings[tf_mask, :] 
            else:
                tf_embeddings = ta_embeddings

            if self.disable_cat:
                h, c = self.decode_step(embeddings[:batch_size_t, t, :], 
                                        (h[:batch_size_t], c[:batch_size_t]))
            else:
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))
            
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            if self.proof_of_concept:
                for bi in range(batch_size_t):
                    predictions[bi, t, targets[bi, t + 1]] = 1
            else:
                predictions[:batch_size_t, t, :] = preds

        return predictions[sort_ind, :, :]

    def predict_init(self, encoder_out):

        if self.encode_step is not None:
            hiddens = None
            for t in range(encoder_out.size(1)):
                hiddens = self.encode_step(encoder_out[:, t, :], hiddens)
            encoder_out = hiddens[0]

        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        return {
            "encoder_out": encoder_out,
            "h": h,
            "c": c,
        }

    def predict_next(self, inputs, contexts):
        
        embeddings = self.embedding(inputs[:, -1])

        if self.disable_cat:
            h, c = self.decode_step(embeddings, (contexts["h"], contexts["c"]))
        else:
            if self.enable_attention:
                attention_weighted_encoding, _ = self.attention(contexts["encoder_out"], contexts["h"])
                gate = self.sigmoid(self.f_beta(contexts["h"]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
            else:
                attention_weighted_encoding = contexts["encoder_out"]

            # with torch.no_grad() should be called outside this scope.
            h, c = self.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), 
                                    (contexts["h"], contexts["c"]))

        scores = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

        predicts = torch.argmax(torch.softmax(scores, dim=-1), dim=-1)
        
        return predicts, scores, {
            "h": h,
            "c": c,
        }


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, 
                 attention_dim, 
                 embed_dim, 
                 decoder_dim, 
                 vocab_size, 
                 max_len_lt,
                 encoder_dim=2048, 
                 dropout=0.5,
                 proof_of_concept: bool = False, 
                 enable_attention = False, 
                 enable_fc = False,
                 enable_encoder = None,
                 enable_memory = None,
                 generator = None,
                 tf_decoder = 1.0,
                 tf_token_decoder = 1.0,
                 ignore_control_tokens = False,
                 disable_cat = False
    ):
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
        self.max_len_lt = max_len_lt
        self.dropout = dropout
        self.proof_of_concept = proof_of_concept
        self.generator = generator
        self.enable_encoder = enable_encoder
        self.enable_memory = enable_memory
        self.tf_decoder = tf_decoder
        self.ignore_control_tokens = ignore_control_tokens

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        if enable_fc:
            self.fc2 = nn.Linear(decoder_dim, decoder_dim)
        else:
            self.fc2 = None
        self.token_decoder = TokenDecoder(embed_dim, decoder_dim, 256, attention_dim, 
                                          vocab_size, dropout=dropout,
                                          proof_of_concept=self.proof_of_concept, 
                                          enable_attention=enable_attention,
                                          enable_encoder=enable_encoder,
                                          tf_token_decoder=tf_token_decoder,
                                          disable_cat=disable_cat)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        if self.fc2 is not None:
            self.fc2.bias.data.fill_(0)
            self.fc2.weight.data.uniform_(-0.1, 0.1)

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
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths, caption_lt_lengths, captions_target):
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

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]
        captions_target = captions_target[sort_ind]
        if self.training:
            caption_lt_lengths = caption_lt_lengths[sort_ind]

        # Embedding
        if self.training:
            embeddings = self.embedding(captions[:, :, 0])  
            # (batch_size, max_caption_length, max_caption_lt_length, embed_dim)
        else:
            embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)
        # print(embeddings.shape)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        if self.training:
            decode_lt_lengths = caption_lt_lengths.flatten().tolist()

        if self.training:
            # Create tensors to hold word predicion scores and alphas
            predictions = torch.zeros(batch_size, max(decode_lengths), max(decode_lt_lengths), vocab_size).to(encoder_out.device)
        else:
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        if self.enable_encoder is not None:
            memory = []
        elif self.enable_memory is not None:
            memory = None

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            alphas[:batch_size_t, t, :] = alpha

            if self.training and t != 0:
                tf_preds = torch.argmax(torch.nn.functional.softmax(
                    predictions[:batch_size_t, t - 1, 0, :], dim=-1), dim=-1)
                tf_embeddings = self.embedding(tf_preds)
                ta_embeddings = embeddings[:batch_size_t, t, :]
                tf_mask = (torch.rand(batch_size_t) < self.tf_decoder)
                tf_embeddings[tf_mask, :] = ta_embeddings[tf_mask, :] 
            else:
                tf_embeddings = embeddings[:batch_size_t, t, :]

            h, c = self.decode_step(
                torch.cat([tf_embeddings, attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc1(self.dropout(h))  # (batch_size_t, vocab_size)

            if self.training:
                if self.proof_of_concept:
                    for bi in range(batch_size_t):
                        predictions[bi, t, 0, captions[bi, t + 1, 0]] = 1
                else:
                    predictions[:batch_size_t, t, 0, :] = preds
            else:
                if self.proof_of_concept:
                    for bi in range(batch_size_t):
                        predictions[bi, t, captions[bi, t + 1]] = 1
                else:
                    predictions[:batch_size_t, t, :] = preds
                
                preds_captions = torch.argmax(torch.nn.functional.softmax(
                    predictions[:batch_size_t, t, :], dim=-1), dim=-1)
                # print(preds_captions.shape, batch_size_t)

            if self.training:
                if self.ignore_control_tokens:
                    indices = torch.where(captions[:batch_size_t, t + 1, 0] > 7)[0]

                    ct_indices = torch.where(torch.logical_and(
                        captions[:batch_size_t, t + 1, 0] <= 7,
                        captions[:batch_size_t, t + 1, 0] != 4
                    ))[0]
                    for ci in ct_indices:
                        predictions[ci, t, 1, 4] = 1  # <end>
                else:
                    indices = torch.where(captions[:batch_size_t, t + 1, 0] != 4)[0]
            else:
                if self.ignore_control_tokens:
                    indices = torch.where(preds_captions > 7)[0]
                else:
                    indices = torch.where(preds_captions != 4)[0]
            if indices.size(0) == 0:
                continue

            if self.fc2 is not None:
                ph = self.fc2(self.dropout(h))
            else:
                ph = h

            if self.enable_encoder is not None:
                if len(memory) > self.enable_encoder - 1 and len(memory) != 0:
                    memory.pop()
                for i in range(len(memory)):
                    memory[i] = memory[i][:batch_size_t, :, :]
                memory.append(ph[:, None, :])
                ph = torch.cat(memory, dim=1)
            elif self.enable_memory is not None:
                if memory is not None:
                    ph = memory[:batch_size_t, :] * self.enable_memory + (1 - self.enable_memory) * ph
                memory = ph 

            if self.training:
                preds_tokens = self.token_decoder(
                    ph[indices], 
                    captions[indices, t + 1, :], 
                    caption_lt_lengths[indices, t + 1])
                predictions[indices, t, 1:1 + preds_tokens.size(1), :] = preds_tokens
            else:
                generator = self.generator(max_seq_len=self.max_len_lt - 1)
                for bi in indices:
                    _, out_scores, out_length = generator.search(
                        self.token_decoder, ph[bi][None], 
                        init_input=preds_captions[bi])
                    if self.proof_of_concept:
                        out_scores = torch.zeros_like(out_scores)
                        out_scores[0, out_length[0] - 2, captions_target[bi, t + 1]] = 1
                    predictions[bi, t, :] = out_scores[0, out_length[0] - 2, :]

            # print(predictions.shape, preds.shape)

        if self.training:
            return predictions, captions, caption_lt_lengths, alphas, sort_ind
        return predictions, captions, decode_lengths, alphas, sort_ind

    def predict(self, encoder_out, captions, hiddens):
        # Embedding
        # print(captions.shape)
        embeddings = self.embedding(captions)  # (batch_size, embed_dim)
        # print(embeddings.shape)

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


class ImageCaptionWithRnn(nn.Module):

    def __init__(
            self, 
            resnet, 
            vocab_size: int, 
            max_len: int,
            max_len_lt: int,
            enable_attention = None, 
            enable_fc = None, 
            enable_encoder = None,
            enable_memory = None,
            disable_cat = None,
            tf_decoder = None,
            tf_token_decoder = None,
            generator=None,
            ignore_control_tokens = None,
            ignore_easy_sample = None,
            proof_of_concept: bool = False
    ):
        super().__init__()

        self.enable_attention = (enable_attention == "1")
        self.enable_fc = (enable_fc == '1')
        self.enable_encoder = (int(enable_encoder) if enable_encoder is not None else None)
        self.enable_memory = (float(enable_memory) if enable_memory is not None else None)
        self.tf_decoder = (float(tf_decoder) if tf_decoder is not None else 1.0)
        self.tf_token_decoder = (float(tf_token_decoder) if tf_token_decoder is not None else 1.0)
        self.ignore_control_tokens = (ignore_control_tokens == '1')
        self.disable_cat = (disable_cat == '1')
        self.ignore_easy_sample = (float(ignore_easy_sample) if ignore_easy_sample is not None else None)
        print("[params] {}".format(", ".join([
            f"{k}={v}" for k, v in {
                "enable_attention": self.enable_attention,
                "enable_fc": self.enable_fc,
                "enable_encoder": self.enable_encoder,
                "enable_memory": self.enable_memory,
                "tf_decoder": self.tf_decoder,
                "tf_token_decoder": self.tf_token_decoder,
                "ignore_control_tokens": self.ignore_control_tokens,
                "disable_cat": self.disable_cat,
                "ignore_easy_sample": self.ignore_easy_sample,
            }.items()
        ])))

        self.proof_of_concept = proof_of_concept
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.alpha_c = 1.
        self.encoder = Encoder(resnet)
        self.decoder = DecoderWithAttention(attention_dim=512,
                                            embed_dim=512,
                                            decoder_dim=512,
                                            vocab_size=vocab_size,
                                            max_len_lt=max_len_lt,
                                            dropout=0.5,
                                            proof_of_concept=self.proof_of_concept,
                                            enable_attention=self.enable_attention,
                                            enable_fc=self.enable_fc,
                                            enable_encoder=self.enable_encoder,
                                            enable_memory=self.enable_memory,
                                            generator=generator,
                                            tf_decoder=self.tf_decoder,
                                            tf_token_decoder=self.tf_token_decoder,
                                            ignore_control_tokens=self.ignore_control_tokens,
                                            disable_cat=self.disable_cat)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, batch):

        if self.proof_of_concept:
            print("### proof_of_concept ###")

        imgs = batch["image"]
        caplens = batch["code_len"]

        if self.training:
            caps = batch["code_train"].long()
            capltlens = batch["code_lt_len"]
            caps_tgt = caps
        else:
            caps = batch["code_valid"].long()
            capltlens = None
            caps_tgt = batch["code"].long()

            # print(caps[caps != 0])
            # print(caps_tgt[caps_tgt != 0])

        batch_size = caps.size(0)
        seq_len = caps.size(1)

        # Forward prop.
        imgs = self.encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            imgs, caps, caplens, capltlens, caps_tgt)

        if self.training:
            targets = caps_sorted[:, 1:]
        else:
            targets = batch["code"].long()
            targets = targets[sort_ind, 1:]

        # print(scores.shape)  # (batch_size, <seq_len, <seq_lt_len, vocab_size)
        # print(targets.shape)  # (batch_size, seq_len-1, seq_lt_len)
        # print(decode_lengths.shape)  # (batch_size, seq_len)

        if self.training:
            xx, yy = [], []
            for bi in range(batch_size):
                for si in range(seq_len - 1):
                    if self.training:
                        dl = decode_lengths[bi, si + 1]
                        if dl == 0:
                            continue
                        # print(bi, si, dl)
                        for di in range(dl):
                            ss = scores[bi, si, di, :]
                            tt = targets[bi, si, di]

                            if self.ignore_easy_sample is not None:
                                sm = torch.nn.functional.softmax(ss, dim=-1)
                                if sm[tt] > self.ignore_easy_sample:
                                    continue

                            xx.append(ss)
                            yy.append(tt)
                            # assert xx[-1].argmax(dim=-1) == yy[-1]
                        # print(torch.argmax(xx[-1], dim=-1), yy[-1])

            scores = torch.stack(xx)
            targets = torch.stack(yy)
        else:
            scores = nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            if self.proof_of_concept:
                print(torch.argmax(scores, dim=-1))
                print(targets)
            # dl = decode_lengths[bi]
            # if dl == 0:
            #     continue
            # # assert dl >= 2, dl
            # # for di in range(dl - 1):
            # #     pp = torch.argmax(scores[bi, si, di + 1, :], dim=-1)
            # #     if pp == 4 or pp == 0:  # <end> or <pad>
            # #         # print(f"found @ {di}")
            # #         break
            # xx.append(scores[bi, si])
            # yy.append(targets[bi, si])
            # # xx.append(scores[bi, si, 0, :])
            # # yy.append(targets[bi, si, 0])

        
        # print(scores.shape, targets.shape)


        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # scores = nn.utils.rnn.pack_padded_sequence(c, d, batch_first=True).data
        # targets = nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = self.criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

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

    def predict(self, images, max_seq_len: int):
        self.eval()

        batch_size = images.size(0)
        # print("batch_size:", batch_size)

        encoder_out = self.encoder(images)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        caps = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(images.device)
        caps[:, 0] = 3  # <start>

        h, c = self.decoder.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        mask = torch.ones((batch_size, ), dtype=torch.bool)
        # print("mask:", mask.shape)

        for k in range(1, max_seq_len):
            if mask.sum() == 0:
                break

            selected_outs = encoder_out[mask]
            selected_caps = caps[mask]
            selected_h = h[mask]
            selected_c = c[mask]

            # print("encoder_out:", encoder_out.shape)
            # print("selected_outs:", selected_outs.shape)
            # print("selected_h:", selected_h.shape)
            # print("selected_c:", selected_c.shape)

            preds, _, hiddens = self.decoder.predict(selected_outs, selected_caps, (selected_h, selected_c))

            outs = torch.argmax(torch.softmax(preds, dim=-1), dim=-1)
            caps[mask, k] = outs

            h[mask] = hiddens[0]
            c[mask] = hiddens[1]

            wh = torch.where(torch.logical_or(caps[:, k] == 0, caps[:, k] == 4))  # <pad> or <end>
            mask[wh] = 0

        return caps

    def predict_beam(self, images, max_seq_len: int, beam_width: int):
        if self.proof_of_concept:
            print("- beam_width:", beam_width)

        self.eval()
        
        batch_size = images.size(0)

        # print("batch_size:", batch_size)

        encoder_out = self.encoder(images)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.repeat_interleave(beam_width, dim=0)  # (batch_size * beam_width, num_pixels, encoder_dim)

        h, c = self.decoder.init_hidden_state(encoder_out)  # (batch_size * beam_width, decoder_dim)

        beam_scorer = transformers.generation.BeamSearchScorer(batch_size, beam_width, images.device, 
                                                               max_length=max_seq_len)

        beam_scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=images.device)
        beam_scores = beam_scores.view(-1)  # (batch_size * beam_width, )

        input_ids = torch.full((batch_size * beam_width, 1), 3, dtype=torch.long, device=images.device)
            
        cur_len = 1
        while cur_len < max_seq_len - 1:
            preds, _, (h, c) = self.decoder.predict(encoder_out, input_ids, (h, c))
            scores = torch.nn.functional.log_softmax(preds, dim=-1)  # (batch_size * beam_width, vocab_size)

            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(batch_size, beam_width * self.vocab_size)

            next_scores, next_tokens = torch.topk(next_scores, 2 * beam_width, dim=1, largest=True, sorted=True)
            next_tokens = next_tokens % self.vocab_size
            next_indices = torch.div(next_tokens, self.vocab_size, rounding_mode="floor")

            # print(next_scores, next_tokens, next_indices)

            r = beam_scorer.process(input_ids, next_scores, next_tokens, next_indices, 
                                    pad_token_id=0, eos_token_id=4)
            # print(r)

            beam_scores = r["next_beam_scores"]
            beam_next_tokens = r["next_beam_tokens"]
            beam_idx = r["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            h = h[beam_idx, :]
            c = c[beam_idx, :]

            if beam_scorer.is_done:
                break

            cur_len += 1

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=0,
            eos_token_id=4,
            max_length=max_seq_len
        )

        return sequence_outputs["sequences"]
    
