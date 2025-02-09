from typing import *

import h5py
import numpy as np
from PIL import Image, ImageDraw

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

    def __init__(self, image_path: str, code_path: str, split: Optional[Any], transform: Optional[Any] = None, 
                 label_trans = None, multi_label: bool = False, label_aug_prob: float = None,
                 has_comma: bool = True, has_rect: bool = False, mask_rect: bool = False):
        super().__init__()
        self.image_path = image_path
        self.code_path = code_path
        self.split = split
        self.transform = transform

        self.multi_label = multi_label
        self.label_aug_prob = label_aug_prob
        
        self.label_trans = label_trans
        if self.label_trans is not None:
            self.max_len_lt = max([len(each) for each in label_trans]) + 1

        assert not (has_comma and has_rect)
        self.has_comma = has_comma
        self.has_rect = has_rect
        self.mask_rect = mask_rect
        self.normalize_rect = True
        
        self.hi = h5py.File(image_path, "r")
        self.images = self.hi["images"]
        self.labels = self.hi["labels"] if has_rect else None  
        self.rects = self.hi["rects"] if has_rect else None
        self.hc = h5py.File(code_path, "r")
        self.max_len = self.hc.attrs["max_len"]
        self.codes = self.hc["ivs"]
        self.code_lens = self.hc["les"]

        self.is_short = self.images.shape[0] < self.codes.shape[0]

        self.idx = self.hc["idx"] if has_rect else None
        self.ids = self.hc["ids"] if has_rect else None
        self.pid = self.hc["pid"] if has_rect else None
        self.piv = self.hc["piv"] if has_rect else None
        
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
        code_idx = self.__idx(index)
        img_idx = code_idx
        if self.has_rect and self.is_short:
            img_idx = self.idx[img_idx]

        image = torch.from_numpy(self.images[img_idx])
        if self.transform is not None:
            image = self.transform(image)

        code = self.codes[code_idx]

        # TODO: the has_comma behavior changes to remove commas from code.
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
            code_len = self.code_lens[code_idx]
            item = {
                "image": image,
                "code": code,
                "code_len": code_len,
            }

        if self.label_trans is not None:
            if self.label_aug_prob is not None:
                for i in range(1, item["code_len"] - 1):
                    if np.random.rand() < self.label_aug_prob:
                        item["code"][i] = np.random.choice(self.label_trans[item["code"][i]])

            if self.multi_label:
                new_code_train = np.zeros((self.max_len, self.max_len_lt), dtype=np.int32)
                new_code_valid = np.zeros((self.max_len, ), dtype=np.int32)
                new_code_lt_len = np.zeros((self.max_len, ), dtype=np.int32)
                for i in range(item["code_len"]):
                    lts = self.label_trans[code[i]][::-1]
                    new_code_train[i, :len(lts)] = lts
                    new_code_train[i, len(lts)] = 4  # <eos>
                    new_code_valid[i] = new_code_train[i, 0] 
                    new_code_lt_len[i] = len(lts) + 1
                item["code_train"] = new_code_train
                item["code_valid"] = new_code_valid
                item["code_lt_len"] = new_code_lt_len 

        if self.has_rect:
            #
            if not self.mask_rect:
                rects = np.stack((np.zeros_like(code, dtype=np.float32), ) * 4, axis=-1)
                ids = self.ids[code_idx]
                ivs = self.codes[code_idx]
                for i, (each_id, each_iv) in enumerate(zip(ids, ivs)):
                    if each_iv <= 7:
                        continue
                    loc = np.where(np.logical_and(
                        self.labels[:, 0] == img_idx,
                        self.labels[:, 1] == each_id
                    ))
                    assert len(loc[0]) == 1, item["code"]
                    rects[i] = self.rects[loc[0]]
                if self.normalize_rect:
                    item["rect"] = box_xyxy_to_cxcywh(rects) / image.size(-1)
                else:
                    item["rect"] = rects
            else:
                masks = np.zeros((code.shape[0], 256, 256), dtype=np.float32)
                ids = self.ids[code_idx]
                ivs = self.codes[code_idx]
                for i, (each_id, each_iv) in enumerate(zip(ids, ivs)):
                    if each_iv <= 7:
                        continue
                    loc = np.where(np.logical_and(
                        self.labels[:, 0] == img_idx,
                        self.labels[:, 1] == each_id
                    ))
                    assert len(loc[0]) == 1, item["code"]
                    rect = self.rects[loc[0]]
                    
                    mask = Image.new("L", (image.size(1), image.size(2)), 0)
                    mask_draw = ImageDraw.Draw(mask)
                    mask_draw.rectangle(rect, fill=255)

                    masks[i] = torch.FloatTensor(np.asarray(mask) / 255.)
                item["mask"] = masks
            #
            if self.is_short:
                mask = Image.new("L", (image.size(1), image.size(2)), 0)

                pid = self.pid[code_idx]
                piv = self.piv[code_idx]
                if pid != -1:
                    loc = np.where(np.logical_and(
                        self.labels[:, 0] == img_idx,
                        self.labels[:, 1] == pid
                    ))
                    assert len(loc[0]) == 1
                    rect = self.rects[loc[0]]
                else:
                    rect = (0, 0, image.size(1) - 1, image.size(2) - 1)

                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(rect, fill=255)
            
                mask = torch.FloatTensor(np.asarray(mask) / 255.)
                mask = mask.unsqueeze(0)
                item["image"] = torch.cat([item["image"], mask], dim=0)
                item["pid"] = pid
                item["piv"] = piv if pid != -1 else 3
        return item
    