from .pix2code import Pix2Code
from .imagecaption import ImageCaption
from .imagecaptionwithbox import ImageCaptionWithBox
# from .imagecaptionwithrpn import ImageCaptionWithRpn
# from .imagecaptionwithspa import ImageCaptionWithSpa
from .imagecaptionwithrnn import ImageCaptionWithRnn
from .imagecaptionwithmsk import ImageCaptionWithMsk
from .imagecaptionwithtwo import ImageCaptionWithTwo
from .vit2code import Vit2Code
from .imagecaptionwithtnn import ImageCaptionWithTnn


def get_model_class_by_name(name):
    if name == "pix2code":
        return Pix2Code
    elif name == "imagecaption":
        return ImageCaption
    elif name in ["imagecaptionwithbox", "icwb"]:
        return ImageCaptionWithBox
    elif name in ["imagecaptionwithrnn", "icwr"]:
        return ImageCaptionWithRnn
    elif name in ["imagecaptionwithtnn", "icwt"]:
        return ImageCaptionWithTnn
    elif name in ["imagecaptionwithtwo", "icw2"]:
        return ImageCaptionWithTwo
    elif name == "vit2code":
        return Vit2Code
    else:
        return None


__all__ = [
    Pix2Code,
    ImageCaption,
    ImageCaptionWithBox,
    ImageCaptionWithRnn,
    # ImageCaptionWithRpn,
    # ImageCaptionWithSpa,
    ImageCaptionWithMsk,
    ImageCaptionWithTwo,
    Vit2Code,
    ImageCaptionWithTnn
]
