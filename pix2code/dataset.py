from typing import *

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


class ImageCodeDataset(Dataset):

    def __init__(self, image_path: str, code_path: str, split: Optional[Any],
                 transform: Optional[Any] = None, has_comma: bool = True):
        super().__init__()
        self.image_path = image_path
        self.code_path = code_path
        self.split = split
        self.transform = transform
        self.has_comma = has_comma
        
        self.hi = h5py.File(image_path, "r")
        self.images = self.hi["images"]
        self.hc = h5py.File(code_path, "r")
        self.max_len = self.hc.attrs["max_len"]
        self.codes = self.hc["ivs"]
        self.code_lens = self.hc["les"]

    def summary(self):
        print()
        print(f"Samples Count: {len(self):,}")
        print(f"   Max Length: {self.max_len:,}")
        print()

    def __len__(self) -> int:
        if self.split is not None:
            return len(self.split)
        return len(self.codes)
    
    def __idx(self, i: int) -> int:
        if self.split is not None:
            return self.split[i]
        return i

    def __getitem__(self, index: int) -> Dict:
        image = torch.from_numpy(self.images[self.__idx(index)])
        if self.transform is not None:
            image = self.transform(image)

        code = self.codes[self.__idx(index)]
        if not self.has_comma:
            code_wo_comma = np.zeros_like(code)
            code = code[code != 7]
            code_len = len(code[code != 0])
            code_wo_comma[:len(code)] = code
            return {
                "image": image,
                "code": code_wo_comma,
                "code_len": code_len
            }
        # code = torch.from_numpy(self.codes[self.__idx(index)]).float()
        code_len = self.code_lens[self.__idx(index)]
        return {
            "image": image,
            "code": code,
            "code_len": code_len,
        }
