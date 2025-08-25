from typing import *


def get_model_class_by_name(name):
    if name == "pix2code":
        from .pix2code import Pix2Code
        return Pix2Code
    elif name == "imagecaption":
        from .imagecaption import ImageCaption
        return ImageCaption
    elif name in ["imagecaptionwithbox", "icwb"]:
        from .imagecaptionwithbox import ImageCaptionWithBox
        return ImageCaptionWithBox
    elif name in ["imagecaptionwithrnn", "icwr"]:
        from .imagecaptionwithrnn import ImageCaptionWithRnn
        return ImageCaptionWithRnn
    elif name in ["imagecaptionwithtnn", "icwt"]:
        from .imagecaptionwithtnn import ImageCaptionWithTnn
        return ImageCaptionWithTnn
    elif name in ["imagecaptionwithtwo", "icw2"]:
        from .imagecaptionwithtwo import ImageCaptionWithTwo
        return ImageCaptionWithTwo
    elif name in ["ImageCaptionWithBit".lower(), "ibit"]:
        from .imagecaptionwithbit import ImageCaptionWithBit
        return ImageCaptionWithBit
    elif name in ["ImageCaptionWithTre".lower(), "tree"]:
        from .imagecaptionwithtre import ImageCaptionWithTre
        return ImageCaptionWithTre
    elif name in ["ImageCaptionWithPtr".lower(), "icwp"]:
        from .imagecaptionwithptr import ImageCaptionWithPtr
        return ImageCaptionWithPtr
    elif name == "vit2code":
        from .vit2code import Vit2Code
        return Vit2Code
    elif name == "Rag2Code".lower():
        from .rag2code import Rag2Code
        return Rag2Code
    elif name == "vit2tree":
        from .vit2tree import Vit2Tree
        return Vit2Tree
    elif name == "rnn2tree":
        from .rnn2tree import Rnn2Tree
        return Rnn2Tree
    elif name == "Vit2Box".lower():
        from .vit2box import Vit2Box
        return Vit2Box
    elif name == "ten":
        from .ten import TreeEditNet
        return TreeEditNet
    elif name == "ten2":
        from .ten2 import TreeEditNet2
        return TreeEditNet2
    elif name == "ui2box".lower():
        from .ui2box import Ui2Box
        return Ui2Box
    else:
        return None


__all__ = [
    get_model_class_by_name
]
