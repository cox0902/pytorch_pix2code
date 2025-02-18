from .pix2code import Pix2Code
from .imagecaption import ImageCaption
# from .imagecaptionwithbox import ImageCaptionWithBox
# from .imagecaptionwithrpn import ImageCaptionWithRpn
# from .imagecaptionwithspa import ImageCaptionWithSpa
from .imagecaptionwithrnn import ImageCaptionWithRnn
from .imagecaptionwithmsk import ImageCaptionWithMsk
from .imagecaptionwithtwo import ImageCaptionWithTwo
from .vit2code import Vit2Code
from .imagecaptionwithtnn import ImageCaptionWithTnn
from .imagecaptionwithbit import ImageCaptionWithBit
from .imagecaptionwithtre import ImageCaptionWithTre
from .imagecaptionwithptr import ImageCaptionWithPtr


def get_model_class_by_name(name):
    if name == "pix2code":
        return Pix2Code
    elif name == "imagecaption":
        return ImageCaption
    elif name in ["imagecaptionwithbox", "icwb"]:
        return ImageCaption
    elif name in ["imagecaptionwithrnn", "icwr"]:
        return ImageCaptionWithRnn
    elif name in ["imagecaptionwithtnn", "icwt"]:
        return ImageCaptionWithTnn
    elif name in ["imagecaptionwithtwo", "icw2"]:
        return ImageCaptionWithTwo
    elif name in ["ImageCaptionWithBit".lower(), "ibit"]:
        return ImageCaptionWithBit
    elif name in ["ImageCaptionWithTre".lower(), "tree"]:
        return ImageCaptionWithTre
    elif name in ["ImageCaptionWithPtr".lower(), "icwp"]:
        return ImageCaptionWithPtr
    elif name == "vit2code":
        return Vit2Code
    elif name == "Rag2Code".lower():
        from .rag2code import Rag2Code
        return Rag2Code
    else:
        return None


__all__ = [
    Pix2Code,
    ImageCaption,
    # ImageCaptionWithBox,
    ImageCaptionWithRnn,
    # ImageCaptionWithRpn,
    # ImageCaptionWithSpa,
    ImageCaptionWithMsk,
    ImageCaptionWithTwo,
    Vit2Code,
    ImageCaptionWithTnn,
    ImageCaptionWithBit,
    ImageCaptionWithTre,
    ImageCaptionWithPtr
]
