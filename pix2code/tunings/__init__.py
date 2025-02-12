from .clsasrnntuning import ClsAsRnnTuning


def get_ft_model_class_by_name(name):
    if name in ["clsasruntuning", "cart"]:
        return ClsAsRnnTuning
    assert False, name


__all__ = [
    ClsAsRnnTuning
]