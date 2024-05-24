from typing import *

import torch
import torchvision.transforms.v2 as T


class PresetEval:
    def __init__(
            self,
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225),
    ):
        self.transforms = T.Compose([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ])

    def __call__(self, img):
        return self.transforms(img)
    