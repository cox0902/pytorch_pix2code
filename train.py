from typing import *

import argparse
from urllib.parse import urlparse, parse_qs
from functools import partial

import json
import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC

from pix2code.utils import seed_everything
from pix2code.trainer import Trainer
from pix2code.metrics import SimpleMulticlassMetrics, SimpleLossMetrics, AdvMetrics
from pix2code.dataset import ImageCodeDataset
from pix2code.transforms import PresetEval
from pix2code.models import get_model_class_by_name
from pix2code.tunings import get_ft_model_class_by_name
from pix2code.generators import GreedySearch, BeamSearch


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--model-resnet", type=str)
    parser.add_argument("--compat", action="store_true")
    parser.add_argument("--lr-find", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--finetune", type=str)

    parser.add_argument("--opt", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--metric", default="auc", type=str)
    parser.add_argument("--stop-metric", type=str)
    parser.add_argument("--eval-metric", type=str)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--epochs-early-stop", default=10, type=int)
    parser.add_argument("--epochs-adjust-lr", default=4, type=int)
    parser.add_argument("--logit-adjustment-train", type=str)
    parser.add_argument("--logit-adjustment-valid", type=str)

    parser.add_argument("--image-path", type=str)
    parser.add_argument("--split-path", type=str)
    parser.add_argument("--code-path", type=str)
    parser.add_argument("--code-lt-path", type=str)
    parser.add_argument("--test-path", type=str)
    parser.add_argument("--multi-label", type=str)
    parser.add_argument("--label-aug-prob", type=float)

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
    if model_name == "pix2code":
        return False, False, False
    elif model_name == "imagecaption":
        return False, False, False
    elif model_name in ["imagecaptionwithbox", "icwb"]:
        return True, True, False
    elif model_name in ["imagecaptionwithrnn", "icwr"]:
        return False, False, False
    elif model_name in ["imagecaptionwithtnn", "icwt"]:
        return False, False, False
    elif model_name in ["imagecaptionwithmsk", "icwm"]:
        return True, False, True
    elif model_name in ["imagecaptionwithspa", "icws"]:
        return True, False, False
    elif model_name in ["vit2code"]:
        return False, False, False
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


def main(args):
    print(args)

    generator, seed_worker = seed_everything(args.seed)

    #

    has_rect, norm_rect, mask_rect = check_model(args.model)

    if args.split_path is not None:
        split = np.load(args.split_path)
        split_train = split["train"]
        split_valid = split["valid"]
        split_test = split["test"]
    else:
        split_train, split_valid, split_test = None, None, None

    if args.code_lt_path is not None:
        with open(args.code_lt_path, "r") as input:
            code_lt = json.load(input)
    else:
        code_lt = None

    has_comma = (not args.no_comma)
    
    train_set = ImageCodeDataset(args.image_path, 
                                 args.code_path, 
                                 split_train, 
                                 transform=PresetEval(),
                                 label_trans=code_lt, 
                                 multi_label=args.multi_label, 
                                 label_aug_prob=args.label_aug_prob,
                                 has_comma=has_comma, 
                                 has_rect=has_rect, 
                                 mask_rect=mask_rect)
    train_set.normalize_rect = norm_rect
    train_set.summary("> Train set")
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              pin_memory=args.pin_memory, 
                              num_workers=args.workers, 
                              worker_init_fn=seed_worker, 
                              generator=generator)
         
    if split_valid is not None:
        valid_set = ImageCodeDataset(args.image_path, 
                                     args.code_path, 
                                     split_valid, 
                                     transform=PresetEval(),
                                     label_trans=code_lt, 
                                     multi_label=args.multi_label,
                                     has_comma=has_comma, 
                                     has_rect=has_rect, 
                                     mask_rect=mask_rect)
        valid_set.normalize_rect = norm_rect
        valid_set.summary("> Valid set")
        valid_loader = DataLoader(valid_set, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  pin_memory=args.pin_memory)
    else:
        valid_loader = None

    #

    model = build_model(args, train_set)

    if args.compat:
        model.criterion = nn.CrossEntropyLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    
    #

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      generator=generator,
                      is_ema=args.ema, 
                      use_amp=args.amp)
    trainer.epochs_early_stop = args.epochs_early_stop
    trainer.epochs_adjust_lr = args.epochs_adjust_lr
    trainer.early_stop = args.early_stop

    if args.grad_clip:
        trainer.grad_clip = 1.
        trainer.grad_clip_fn = nn.utils.clip_grad.clip_grad_value_

    if args.lr_find:
        trainer.lr_find(end_lr=100., 
                        step_mode='exp', 
                        epochs=100,
                        train_loader=train_loader, 
                        valid_loader=valid_loader)
        return
    
    if args.metric == "acc":
        metrics = SimpleMulticlassMetrics(90, scorer=MulticlassAccuracy)
    elif args.metric == "auc":
        metrics = SimpleMulticlassMetrics(90, scorer=MulticlassAUROC)
    elif args.metric == "loss":
        metrics = SimpleLossMetrics()
    else:
        assert False

    if args.eval_metric is None:
        eval_metrics = metrics
    else:
        eval_metrics = AdvMetrics(reduction=args.stop_metric)
        ems = args.eval_metric.split("+")
        for each in ems:
            eval_metrics.add_metric(each)

    trainer.fit(epochs=args.epochs, 
                train_loader=train_loader, 
                valid_loader=valid_loader, 
                metrics=metrics, 
                eval_metrics=eval_metrics, 
                proof_of_concept=args.proof_of_concept)
    
    if split_test is not None:
        print("=" * 100)
        test_set = ImageCodeDataset(args.image_path, 
                                    args.test_path, 
                                    split_test, 
                                    transform=PresetEval(),
                                    label_trans=code_lt, 
                                    multi_label=args.multi_label,
                                    has_comma=has_comma, 
                                    has_rect=has_rect, 
                                    mask_rect=mask_rect)
        test_set.normalize_rect = norm_rect
        test_set.summary("> Test set")
        test_loader = DataLoader(test_set, 
                                 batch_size=args.batch_size, 
                                 shuffle=False, 
                                 pin_memory=args.pin_memory)
        
        trainer = Trainer.load_checkpoint("./BEST.pth.tar")
        _ = trainer.test(data_loader=test_loader, 
                         metrics=eval_metrics, 
                         proof_of_concept=args.proof_of_concept)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)