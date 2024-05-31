from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sort_n_pack_padded_sequence, pad_packed_sequence_n_unsort


class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            # [batch_size, 3, 256, 256]
            nn.Conv2d(3, 32, 3),  # [batch_size, 32, 254, 254]
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),  # [batch_size, 32, 252, 252]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch_size, 32, 126, 126] 
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),  # [batch_size, 64, 124, 124]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),  # [batch_size, 64, 122, 122]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch_size, 64, 61, 61]
            nn.Dropout(0.25)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),  # [batch_size, 128, 59, 59]
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),  # [batch_size, 128, 57, 57]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch_size, 128, 28, 28]
            nn.Dropout(0.25)
        )
        self.layer4 = nn.Sequential(
            nn.Flatten(),  # [batch_size, 128 * 28 * 28]
            nn.Linear(128 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),  # [batch_size, 1024]
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [batch_size, 1024]
    

class ContextEncoder(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(vocab_size, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True)

    def forward(self, x: torch.Tensor, x_len, h: Optional[Tuple] = None) -> torch.Tensor:
        # x = [batch_size, seq_length, vocab_size]

        if h is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                             dtype=x.dtype, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                             dtype=x.dtype, device=x.device)
            h = (h0, c0)

        x_packed, idx_unsort = sort_n_pack_padded_sequence(x, x_len)

        y_packed, _ = self.rnn(x_packed, h)

        y = pad_packed_sequence_n_unsort(y_packed, idx_unsort, max_len=x.size(1))
        return y  # [batch_size, seq_length, hidden_size]
    

class Decoder(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(1024 + 128, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x_image: torch.Tensor, x_context: torch.Tensor, x_length,
                h: Optional[Tuple] = None) -> torch.Tensor:
        x_image = x_image.unsqueeze(1)  # [batch_size, 1024] -> [batch_size, 1, 1024]
        x_image = x_image.repeat(1, x_context.size(1), 1)  # -> [batch_size, seq_length, 1024]

        x = torch.cat((x_image, x_context), dim=2)  # -> [batch_size, seq_length, 1024 + 128]

        if h is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                             dtype=x.dtype, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                             dtype=x.dtype, device=x.device)
            h = (h0, c0)

        x_packed, _ = sort_n_pack_padded_sequence(x, x_length)

        y_packed, _ = self.rnn(x_packed, h)  # y_packed.data -> [-1, hidden_size]

        y_packed = self.fc(y_packed.data)
        return y_packed  # softmax is omit for CrossEntropyLoss  
    

class Pix2Code(nn.Module):

    def __init__(self, vocab_size: int):
        super(Pix2Code, self).__init__()
        self.image_encoder = ImageEncoder()
        self.context_encoder = ContextEncoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def forward(self, batch: Dict) -> torch.Tensor:
        # batch["code"] = [batch_size, seq_length]
        context = F.one_hot(batch["code"][:, :-1].long(), num_classes=90).float()  # -> [batch_size, seq_length, vocab_size]

        encoded_image = self.image_encoder(batch["image"])
        context_length = batch["code_len"] - 1
        encoded_context = self.context_encoder(context, context_length)
        decoded = self.decoder(encoded_image, encoded_context, context_length)  # -> [-1, vocab_size]

        target_packed, _ = sort_n_pack_padded_sequence(batch["code"][:, 1:].long(), context_length)
        return decoded, F.softmax(decoded, dim=-1), target_packed.data