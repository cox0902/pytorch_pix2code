from .pix2code import Pix2Code
from .imagecaption import ImageCaption
from .imagecaptionwithbox import ImageCaptionWithBox
from .detr import Detr


__all__ = [
    Pix2Code,
    ImageCaption,
    ImageCaptionWithBox,
    Detr
]

# fix torch 2.4.0 warning

import torch

torch.serialization.add_safe_globals(__all__)