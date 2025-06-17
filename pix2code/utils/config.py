from typing import *

import argparse
from urllib.parse import urlparse, parse_qs
from functools import partial

import numpy as np

import torchvision

from ..models import get_model_class_by_name
from ..tunings import get_ft_model_class_by_name
from ..generators import GreedySearch, BeamSearch
from ..trainer import Trainer


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model-resnet", type=str)
    parser.add_argument("--compat", action="store_true")
    parser.add_argument("--lr-find", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--finetune", type=str)
    parser.add_argument("--test-only", action="store_true", default=False)

    parser.add_argument("--opt", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--metric", default="auc", type=str)
    parser.add_argument("--stop-metric", type=str)
    parser.add_argument("--eval-metric", type=str)
    parser.add_argument("--early-stop", action="store_true", default=False)
    parser.add_argument("--epochs-early-stop", default=10, type=int)
    parser.add_argument("--epochs-adjust-lr", default=4, type=int)
    parser.add_argument("--logit-adjustment-train", type=str)
    parser.add_argument("--logit-adjustment-valid", type=str)

    parser.add_argument("--retriever-fn", type=str)
    parser.add_argument("--retriever-index-path", type=str)
    parser.add_argument("--retriever-image-path", type=str)
    parser.add_argument("--retriever-code-path", type=str)
    parser.add_argument("--retriever-split-path", type=str)

    parser.add_argument("--image-path", type=str)
    parser.add_argument("--split-path", type=str)
    parser.add_argument("--code-path", type=str)
    parser.add_argument("--code-lt-path", type=str)
    parser.add_argument("--test-path", type=str)
    parser.add_argument("--multi-label", type=str)
    parser.add_argument("--label-aug-prob", type=float)
    parser.add_argument("--force-add-channel", action="store_true", default=False)
    parser.add_argument("--has-tree", action="store_true", default=False)
    parser.add_argument("--mask-tree", action="store_true", default=False)
    parser.add_argument("--force-valid", action="store_true", default=False)
    parser.add_argument("--nest-rect", action="store_true", default=False)

    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--pin-memory", action="store_true")
    
    parser.add_argument("--grad-clip", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", default=3600, type=int)

    parser.add_argument("--no-comma", action="store_true")

    parser.add_argument("--extra", type=str)

    return parser


def parse_model(model: str):
    # TODO: Overwrite preset params.
    model_name = model
    model_params = {}
    if model.startswith("vm://"):
        model_url = urlparse(model)
        model_name = model_url.netloc
        model_params = parse_qs(model_url.query)
        model_params = { k: v[0] for k, v in model_params.items() }
    return model_name, model_params


def build_resnet_model(model_resnet: str, verbose: bool = True):
    model_resnet_name, model_resnet_params = parse_model(model_resnet)
    if model_resnet_name == "resnet":
        variant = model_resnet_params["variant"]
        load_weight = ("load_weight" in model_resnet_params) and (model_resnet_params["load_weight"] != "0")
        if verbose:
            print(f">>> Build {model_resnet_name} with variant={variant} and load_weight={load_weight}")
        if variant == "50":
            if load_weight:
                return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            else:
                return torchvision.models.resnet50()
        elif variant == "101":
            if load_weight:
                return torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
            else:
                return torchvision.models.resnet101()
        else:
            assert False
    elif model_resnet_name == "resnext":
        variant = model_resnet_params["variant"]
        load_weight = ("load_weight" in model_resnet_params) and (model_resnet_params["load_weight"] != "0")
        if verbose:
            print(f">>> Build {model_resnet_name} with variant={variant} and load_weight={load_weight}")
        if variant == "50":
            if load_weight:
                return torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
            else:
                return torchvision.models.resnext50_32x4d()
        else:
            assert False
    elif model_resnet_name == "vis":
        variant = model_resnet_params["variant"]
        load_weight = ("load_weight" in model_resnet_params) and (model_resnet_params["load_weight"] != "0")
        copy_weight = ("copy_weight" in model_resnet_params) and (model_resnet_params["copy_weight"] != "0")
        if verbose:
            print(f">>> Build {model_resnet_name} with variant={variant}, load_weight={load_weight} and copy_weight={copy_weight}")
        from dt.models import VisModel
        return VisModel(model=variant, load_weight=load_weight, copy_weight=copy_weight).resnet
    else:
        if verbose:
            print(f">>> Build vis from checkpoint")
        from dt.trainer import Trainer as DtTrainer
        t = DtTrainer.load_checkpoint(model_resnet_name)
        return t.get_inner_model().resnet


def check_model(model: str) -> Tuple[bool, bool]:
    # returns (has_rect, norm_rect, mask_rect)
    model_name, model_params = parse_model(model)
    assert model_name is not None
    # print(model_name)
    if model_name in ["imagecaptionwithbox", "icwb"]:
        return True, False, False
    elif model_name in ["vit2box"]:
        return True, True, False
    elif model_name in ["imagecaptionwithmsk", "icwm"]:
        return True, False, True
    elif model_name in ["imagecaptionwithspa", "icws"]:
        return True, False, False
    else:
        return False, False, False


def build_model(args, data_set):
    model_name, model_params = parse_model(args.model)
    assert model_name is not None

    model_class = get_model_class_by_name(model_name)
    if model_class is None:
        t = Trainer.load_checkpoint(model_name)
        m = t.get_inner_model()
        if args.finetune is not None:
            ft_model_name, ft_model_params = parse_model(args.finetune)
            ft_model_class = get_ft_model_class_by_name(ft_model_name)
            m = ft_model_class(m, **ft_model_params)
        return m

    if args.proof_of_concept:
        model_params["proof_of_concept"] = True

    model_resnet = args.model_resnet
    if model_resnet is not None:
        model_params["resnet"] = build_resnet_model(model_resnet)
    
    model_params["vocab_size"] = 90
    
    model_params["max_len"] = data_set.max_len
    
    if args.code_lt_path is not None:
        model_params["max_len_lt"] = data_set.max_len_lt

    if args.retriever_index_path is not None:
        # split = None
        # if args.retriever_split_path is not None:
        #     split = np.load(args.retriever_split_path)["train"]
        # import faiss
        from pix2code.models.rag2code import retrieve_fn, retrieve_fn_old

        fn = retrieve_fn
        if args.retriever_fn == "old":
            fn = retrieve_fn_old

        # index = faiss.read_index(args.retriever_index_path)
        # database = ImageCodeDataset(args.retriever_image_path,
        #                             args.retriever_code_path,
        #                             split)

        model_params["retrieve_fn"] = partial(fn, 
                                              index_path=args.retriever_index_path, 
                                              image_path=args.retriever_image_path,
                                              code_path=args.retriever_code_path)

    if "generator" in model_params:
        if model_params["generator"].startswith("beam"):
            model_params["generator"] = partial(
                BeamSearch, 
                vocab_size=model_params["vocab_size"], 
                beam_width=int(model_params["generator"][-1]))
        else:
            emb_weight = np.load(args.extra) if args.extra is not None else None
            model_params["generator"] = partial(
                GreedySearch, 
                vocab_size=model_params["vocab_size"], 
                conditions=emb_weight)

    return model_class(**model_params)
