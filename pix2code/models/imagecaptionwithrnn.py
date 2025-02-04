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

    def __init__(self, embed_dim, encoder_dim, decoder_dim, vocab_size, dropout=0.2, proof_of_concept=False, 
                 emb_weight=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.proof_of_concept = proof_of_concept

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        
        if emb_weight is not None:
            tx = torch.from_numpy(emb_weight)
            tx.clamp_min_(0.01)
            self.emb_weight = nn.Embedding(90, 90, _weight=tx, _freeze=True)
        else:
            self.emb_weight = None

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

        h, c = self.init_hidden_state(encoder_out)

        embeddings = self.embedding(targets)

        decode_lengths = (target_lengths - 1).tolist()

        predictions = torch.zeros(len(decode_lengths), max(decode_lengths), self.vocab_size).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], encoder_out[:batch_size_t]], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            if self.proof_of_concept:
                for bi in range(batch_size_t):
                    predictions[bi, t, targets[bi, t + 1]] = 1
            else:
                if self.emb_weight is not None:
                    preds = preds * self.emb_weight(targets[:batch_size_t, t])
                predictions[:batch_size_t, t, :] = preds

        return predictions, sort_ind

    def predict_init(self, encoder_out):
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        return {
            "encoder_out": encoder_out,
            "h": h,
            "c": c,
        }

    def predict_next(self, inputs, contexts):
        
        embeddings = self.embedding(inputs)

        # with torch.no_grad() should be called outside this scope.
        h, c = self.decode_step(torch.cat([embeddings, contexts["encoder_out"]], dim=1), 
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

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5,
                 proof_of_concept: bool = False, emb_weight = None, generator = None):
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
        self.proof_of_concept = proof_of_concept
        self.generator = generator

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.token_decoder = TokenDecoder(embed_dim, decoder_dim, 256, vocab_size, dropout=dropout,
                                          proof_of_concept=self.proof_of_concept, emb_weight=emb_weight)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

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

    def forward(self, encoder_out, encoded_captions, caption_lengths, caption_lt_lengths):
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
        encoded_captions = encoded_captions[sort_ind]
        caption_lt_lengths = caption_lt_lengths[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions[:, :, 0])  # (batch_size, max_caption_length, embed_dim)
        # print(embeddings.shape)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        decode_lt_lengths = caption_lt_lengths.flatten().tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), max(decode_lt_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            if self.proof_of_concept:
                for bi in range(batch_size_t):
                    predictions[bi, t, 0, encoded_captions[bi, t + 1, 0]] = 1
            else:
                predictions[:batch_size_t, t, 0] = preds
            preds_tokens, sort_tokens = self.token_decoder(h, encoded_captions[:batch_size, t + 1, :], caption_lt_lengths[:batch_size_t, t + 1])
            # print(predictions.shape, preds.shape)
            predictions[sort_tokens, t, 1:1 + preds_tokens.size(1), :] = preds_tokens
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, caption_lt_lengths, alphas, sort_ind

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

    def __init__(self, resnet, vocab_size: int, emb_weight = None, generator=None):
        super().__init__()
        print(f"[params] emb_weight={emb_weight is not None}")

        self.proof_of_concept: bool = False
        self.vocab_size = vocab_size
        self.alpha_c = 1.
        self.encoder = Encoder(resnet)
        self.decoder = DecoderWithAttention(attention_dim=512,
                                            embed_dim=512,
                                            decoder_dim=512,
                                            vocab_size=vocab_size,
                                            dropout=0.2,
                                            proof_of_concept=self.proof_of_concept,
                                            emb_weight=emb_weight,
                                            generator=generator)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, batch):
        imgs = batch["image"]
        caps = batch["code"].long()
        caplens = batch["code_len"]
        capltlens = batch["code_lt_len"]

        batch_size = caps.size(0)
        seq_len = caps.size(1)
        seq_lt_len = caps.size(2)

        # Forward prop.
        imgs = self.encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, caps, caplens, capltlens)

        targets = caps_sorted[:, 1:, :]

        # print(scores.shape)  # (batch_size, <seq_len, <seq_lt_len, vocab_size)
        # print(targets.shape)  # (batch_size, seq_len-1, seq_lt_len)
        # print(decode_lengths.shape)  # (batch_size, seq_len)

        xx, yy = [], []
        for bi in range(batch_size):
            for si in range(seq_len - 1):
                dl = decode_lengths[bi, si + 1]
                if dl == 0:
                    continue
                # print(bi, si, dl)
                for di in range(dl):
                    xx.append(scores[bi, si, di, :])
                    yy.append(targets[bi, si, di])
                    # assert xx[-1].argmax(dim=-1) == yy[-1]

        scores = torch.stack(xx)
        targets = torch.stack(yy)
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
    
