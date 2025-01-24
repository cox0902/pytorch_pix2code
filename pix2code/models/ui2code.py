import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class CNNModel(nn.Module):
    def __init__(self, n_channels=1):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),  # (batch_size, 64, 256, 256)
            nn.ReLU(),

            nn.MaxPool2d(2),  # (batch_size, 64, 128, 128)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (batch_size, 128, 128, 128)
            nn.ReLU(),
            
            nn.MaxPool2d(2),  # (batch_size, 128, 64, 64)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (batch_size, 256, 64, 64)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (batch_size, 256, 64, 64)
            nn.ReLU(),
            
            nn.MaxPool2d((1, 2), stride=(1, 2)),  # (batch_size, 256, 32, 64)
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (batch_size, 512, 32, 64)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.MaxPool2d((2, 1), stride=(2, 1)),  # (batch_size, 512, 32, 32)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (batch_size, 512, 32, 32)
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x).permute(0, 2, 3, 1)  # (batch_size, H, W, C)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, dropout=0.0):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout
        )
        self.bidirectional = bidirectional

    def forward(self, x, hidden=None):
        outputs, hidden = self.lstm(x, hidden)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, num_hidden, simple=False):
        super(Attention, self).__init__()
        self.simple = simple

        self.attn_linear = nn.Linear(num_hidden, num_hidden, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.attn_tanh = nn.Tanh()
        self.context_combine = nn.Linear(num_hidden * 2, num_hidden, bias=False)

    def forward(self, target_t, context):
        # target_t: (batch_size, num_hidden)
        # context: (batch_size, max_seq_len, num_hidden)
        score_t = self.attn_linear(target_t)  # (batch_size, num_hidden)
        attn = torch.bmm(context, score_t.unsqueeze(-1))  # (batch_size, max_seq_len, 1)

        attn = attn.squeeze(-1)  # (batch_size, max_seq_len)
        attn = self.softmax(attn)
        attn = attn.unsqueeze(1)  # (batch_size, 1, max_seq_len)

        context_combined = torch.bmm(attn, context)  # (batch_size, 1, num_hidden)
        context_combined = context_combined.unsqueeze(1)  # (batch_size, num_hidden)

        if not self.simple:
            combined_context = torch.cat([combined_context, target_t], dim=-1)
            context_output = self.attn_tanh(self.context_combine(combined_context))
        else:
            context_output = context_combined + target_t
        return context_output


class LSTMDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, attention, dropout=0.0):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = attention
        self.lstm = nn.LSTM(
            embedding_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden, encoder_outputs):
        input = self.embedding(input).unsqueeze(1)
        attn_weights = self.attention(hidden[0][-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        lstm_input = torch.cat((input, context.unsqueeze(1)), dim=2)
        output, hidden = self.lstm(lstm_input, hidden)
        prediction = self.fc_out(torch.cat((output.squeeze(1), context), dim=1))
        return prediction, hidden, attn_weights


class Seq2SeqModel(nn.Module):
    def __init__(self, cnn_feature_size, encoder_num_hidden, encoder_num_layers, decoder_num_layers, 
                 target_vocab_size, target_embedding_size, max_encoder_l_w, max_encoder_l_h, max_decoder_l,
                 dropout):
        super(Seq2SeqModel, self).__init__()
        decoder_num_hidden = encoder_num_hidden * 2

        self.pos_embedding_fw = nn.Embedding(max_encoder_l_h, encoder_num_layers * encoder_num_hidden * 2)
        self.pos_embedding_bw = nn.Embedding(max_encoder_l_h, encoder_num_layers * encoder_num_hidden * 2)
        self.cnn_model = CNNModel()
        self.encoder_fw = nn.LSTM(
            cnn_feature_size, encoder_num_hidden, encoder_num_layers, batch_first=True, dropout=dropout
        )
        self.encoder_bw = nn.LSTM(
            cnn_feature_size, encoder_num_hidden, encoder_num_layers, batch_first=True, dropout=dropout
        )
        self.decoder = nn.LSTM(
            target_embedding_size, decoder_num_hidden, decoder_num_layers, batch_first=True, dropout=dropout
        )
        self.output_projector = nn.Linear(decoder_num_hidden, target_vocab_size)

        # attention = Attention(hidden_size=decoder_hidden_size)

    def forward(self, input_batch, target_batch=None):
        # CNN Encoder
        cnn_outputs = self.cnn_model(input_batch)
        batch_size, H, W, _ = cnn_outputs.size()
        encoder_outputs_fw = []
        encoder_outputs_bw = []

        # Forward LSTM Encoder
        for h in range(H):
            outputs_fw, _ = self.encoder_fw(cnn_outputs[:, h, :, :])
            encoder_outputs_fw.append(outputs_fw)

        # Backward LSTM Encoder
        for h in range(H):
            outputs_bw, _ = self.encoder_bw(cnn_outputs[:, H - h - 1, :, :])
            encoder_outputs_bw.append(outputs_bw)

        encoder_outputs = torch.cat(
            [torch.cat(encoder_outputs_fw, dim=1), torch.cat(encoder_outputs_bw[::-1], dim=1)], dim=2
        )

        # Decoder
        decoder_hidden = None  # Initialize decoder state
        decoder_outputs = []
        if target_batch is not None:
            for t in range(target_batch.size(1)):
                decoder_input = target_batch[:, t]
                output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs.append(output)
            return torch.stack(decoder_outputs, dim=1)
        else:
            # Inference mode (greedy decoding or beam search can be implemented here)
            return encoder_outputs


# Example Configuration
config = {
    "encoder_num_layers": 2,
    "decoder_num_layers": 2,
    "target_embedding_size": 256,
    "dropout": 0.5,
    "batch_size": 32,
}

# Example Initialization
model = Seq2SeqModel(
    cnn_feature_size=512,
    encoder_hidden_size=512,
    decoder_hidden_size=1024,
    target_vocab_size=10000,
    config=config,
)