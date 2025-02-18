from .clsasrnntuning import ClsAsRnnTuning
from .pointertuning import PointerTuning


def get_ft_model_class_by_name(name):
    if name in ["clsasruntuning", "cart"]:
        return ClsAsRnnTuning
    elif name in ["pointertuning", "poit"]:
        return PointerTuning
    assert False, name


__all__ = [
    ClsAsRnnTuning,
    PointerTuning
]