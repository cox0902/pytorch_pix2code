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

        if pos_embed == '1':
            self.pos_embedding = PositionalEmbedding(max_len, embed_dim)
        else:
            self.pos_embedding = None

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim + 1024, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
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

    def forward(self, encoder_out, encoder_txt, encoded_captions, caption_lengths):
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

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        if hasattr(self, "pos_embedding") and self.pos_embedding is not None:
            embeddings = self.pos_embedding(embeddings)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # print(attention_weighted_encoding.shape)

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], encoder_txt[:batch_size_t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
    def predict(self, encoder_out, captions, hiddens = None):
        # Embedding
        # print(captions.shape)
        embeddings = self.embedding(captions[:, -1])  # (batch_size, embed_dim)
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


class TextEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super().__init__()

        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, embedded_inputs, input_lengths):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_inputs, 
            input_lengths.cpu(), 
            batch_first=self.batch_first,
            enforce_sorted=False)
        # Forward pass through RNN
        outputs, hidden = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=self.batch_first, 
                                                      total_length=embedded_inputs.size(1))
        # Return output and final hidden state
        return outputs, hidden
    

class Rnn2Tree(nn.Module):

    def __init__(self, 
                 resnet, 
                 vocab_size: int, 
                 max_len, 
                 pos_embed: str = None,
                 proof_of_concept = False):
        super().__init__()
        print(f"[params] pos_embed={pos_embed}")
       
        self.proof_of_concept: bool = False
        self.vocab_size = vocab_size
        self.alpha_c = 1.
        self.encoder = Encoder(resnet)
        self.text_encoder = TextEncoder(embedding_dim=512, hidden_size=512, num_layers=1)
        self.decoder = DecoderWithAttention(max_len,
                                            attention_dim=512,
                                            embed_dim=512,
                                            decoder_dim=512,
                                            vocab_size=vocab_size,
                                            dropout=0.5,
                                            pos_embed=pos_embed)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, batch):
        imgs = batch["image"]
        caps = batch["code"].long()
        code_cond = batch["code_cond"].long()
        code_len = batch["code_len"]
        code_cond_len = batch["code_cond_len"]

        # Forward prop.
        imgs = self.encoder(imgs)
        embs_code_cond = self.decoder.embedding(code_cond)
        txts, _ = self.text_encoder(embs_code_cond, code_cond_len)
        txts = txts[:, -1, :]
        # print(txts[0].shape, txts[1][0].shape)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, txts, caps, code_len)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = self.criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        return {
            "loss": loss, 
            "logits": scores,
            "scores": torch.nn.functional.softmax(scores, dim=-1), 
            "targets": targets
        }
    
    def get_alphas(self, batch):
        self.eval()

        imgs = batch["image"]
        caps = batch["code"].long()
        caplens = batch["code_len"]

        # Forward prop.
        imgs = self.encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, caps, caplens)

        return {
            "scores": scores,
            "targets": caps_sorted,
            "alphas": alphas
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
