from typing import *

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = np.split(x, 4, axis=1)
    b = [((x0 + x1) * 0.5), ((y0 + y1) * 0.5),
         (x1 - x0), (y1 - y0)]
    return np.concatenate(b, axis=1)

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = np.split(x, 4, axis=1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return np.concatenate(b, axis=1)


class ImageCodeDataset(Dataset):

    def __init__(self, image_path: str, code_path: str, split: Optional[Any],
                 transform: Optional[Any] = None, has_comma: bool = True, has_rect: bool = False):
        super().__init__()
        self.image_path = image_path
        self.code_path = code_path
        self.split = split
        self.transform = transform
        self.has_comma = has_comma
        self.has_rect = has_rect
        
        self.hi = h5py.File(image_path, "r")
        self.images = self.hi["images"]
        self.labels = self.hi["labels"] if has_rect else None  
        self.rects = self.hi["rects"] if has_rect else None
        self.hc = h5py.File(code_path, "r")
        self.max_len = self.hc.attrs["max_len"]
        self.ids = self.hc["ids"] if has_rect else None
        self.codes = self.hc["ivs"]
        self.code_lens = self.hc["les"]

    def summary(self, header: Optional[str] = None):
        print()
        if header is not None:
            print(header)
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
            item = {
                "image": image,
                "code": code_wo_comma,
                "code_len": code_len
            }
        else:
            # code = torch.from_numpy(self.codes[self.__idx(index)]).float()
            code_len = self.code_lens[self.__idx(index)]
            item = {
                "image": image,
                "code": code,
                "code_len": code_len,
            }
        if self.has_rect:
            ignore = np.zeros_like(code)
            equal = np.zeros_like(code)
            rects = np.stack((np.zeros_like(code, dtype=np.float32), ) * 4, axis=-1)
            for i, each_id in enumerate(self.ids[self.__idx(index)]):
                if each_id <= 0:
                    ignore[i] = 1
                    continue
                loc = np.where(np.logical_and(
                    self.labels[:, 0] == self.__idx(index),
                    self.labels[:, 1] == each_id
                ))
                if len(loc[0]) != 1:
                    equal[i] = 1
                    continue
                rects[i] = self.rects[loc[0]]
            item["rect"] = box_xyxy_to_cxcywh(rects) / image.size(-1)
            item["equal"] = equal
            item["ignore"] = ignore
        return item
    