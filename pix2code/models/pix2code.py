from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import sort_n_pack_padded_sequence, pad_packed_sequence_n_unsort


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
            h = self.init_hidden_state(x)

        x_packed, idx_unsort = sort_n_pack_padded_sequence(x, x_len)

        y_packed, _ = self.rnn(x_packed, h)

        y = pad_packed_sequence_n_unsort(y_packed, idx_unsort, max_len=x.size(1))
        return y  # [batch_size, seq_length, hidden_size]
    
    def init_hidden_state(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        return h, c

    def predict(self, x, hiddens):
        y, hiddens = self.rnn(x, hiddens)
        return y, hiddens
    

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
            h = self.init_hidden_state(x)

        x_packed, _ = sort_n_pack_padded_sequence(x, x_length)

        y_packed, _ = self.rnn(x_packed, h)  # y_packed.data -> [-1, hidden_size]

        y_packed = self.fc(y_packed.data)
        return y_packed, h  # softmax is omit for CrossEntropyLoss  
    
    def init_hidden_state(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        return h, c
    
    def predict(self, x_image: torch.Tensor, x_context, hiddens):
        x_image = x_image.unsqueeze(1)  # (batch_size, 1, 1024)
        # print(x_image.shape, x_context.shape)
        x = torch.cat((x_image, x_context), dim=-1)
        y, hiddens = self.rnn(x, hiddens)
        y = self.fc(y)
        return y.squeeze(1), hiddens
    

class Pix2Code(nn.Module):

    def __init__(self, vocab_size: int):
        super(Pix2Code, self).__init__()
        self.image_encoder = ImageEncoder()
        self.context_encoder = ContextEncoder(vocab_size)
        self.decoder = Decoder(vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: Dict) -> torch.Tensor:
        # batch["code"] = [batch_size, seq_length]
        context = F.one_hot(batch["code"][:, :-1].long(), num_classes=90).float()  # -> [batch_size, seq_length, vocab_size]

        encoded_image = self.image_encoder(batch["image"])
        context_length = batch["code_len"] - 1
        encoded_context = self.context_encoder(context, context_length)
        decoded, _ = self.decoder(encoded_image, encoded_context, context_length)  # -> [-1, vocab_size]

        target_packed, _ = sort_n_pack_padded_sequence(batch["code"][:, 1:].long(), context_length)

        loss = self.criterion(decoded, target_packed.data)
        return {
            "loss": loss, 
            "scores": F.softmax(decoded, dim=-1), 
            "targets": target_packed.data
        }
    
    def predict_init(self, images):
        encoded_image = self.image_encoder(images)

        h_en, c_en = self.context_encoder.init_hidden_state(images)
        h_de, c_de = self.decoder.init_hidden_state(images)

        return {
            "encoded_image": encoded_image,
            "h_en": h_en.permute(1, 0, 2),
            "c_en": c_en.permute(1, 0, 2),
            "h_de": h_de.permute(1, 0, 2),
            "c_de": c_de.permute(1, 0, 2)
        }
    
    def predict_next(self, inputs, contexts):

        inputs = F.one_hot(inputs.long(), num_classes=90).float().unsqueeze(1)

        h_en = contexts["h_en"].permute(1, 0, 2)
        c_en = contexts["c_en"].permute(1, 0, 2)
        encoded_context, (h_en, c_en) = self.context_encoder.predict(inputs, (h_en, c_en))

        h_de = contexts["h_de"].permute(1, 0, 2)
        c_de = contexts["c_de"].permute(1, 0, 2)
        scores, (h_de, c_de) = self.decoder.predict(contexts["encoded_image"], encoded_context, (h_de, c_de))
        
        # print(scores.shape)
        predicts = torch.argmax(torch.softmax(scores, dim=-1), dim=-1)

        # print(predicts.shape)

        return predicts, scores, {
            "h_en": h_en.permute(1, 0, 2),
            "c_en": c_en.permute(1, 0, 2),
            "h_de": h_de.permute(1, 0, 2),
            "c_de": c_de.permute(1, 0, 2),
        }