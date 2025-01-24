from typing import *

import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC

from pix2code.utils import seed_everything
from pix2code.trainer import Trainer
from pix2code.metrics import SimpleMulticlassMetrics
from pix2code.dataset import ImageCodeDataset
from pix2code.transforms import PresetEval
from pix2code.models import Pix2Code, ImageCaption, ImageCaptionWithBox, Detr


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--model-resnet", type=str)
    parser.add_argument("--compat", action="store_true")
    parser.add_argument("--lr-find", action="store_true")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--opt", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--metric", default="auc", type=str)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--epochs-early-stop", default=10, type=int)
    parser.add_argument("--epochs-adjust-lr", default=4, type=int)

    parser.add_argument("--image-path", type=str)
    parser.add_argument("--split-path", type=str)
    parser.add_argument("--code-path", type=str)
    parser.add_argument("--test-path", type=str)

    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    
    parser.add_argument("--grad-clip", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", default=3600, type=int)

    parser.add_argument("--no-comma", action="store_true")

    return parser


def main(args):
    print(args)

    generator, seed_worker = seed_everything(args.seed)

    if args.model == "pix2code":
        model = Pix2Code(vocab_size=90)
    elif args.model == 'imagecaption':
        if args.model_resnet is None:
            resnet = None
        elif args.model_resnet == "50":
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        elif args.model_resnet == "101":
            resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        elif args.model_resnet.startswith("vm://"):
            model_url = urlparse(args.model_resnet)
            model_name = model_url.netloc
            model_params = parse_qs(model_url.query)
            model_variant = model_params["variant"][0]
            print(f"vm:// {model_name} ? variant={model_variant}")
            if model_name == "resnet":
                if model_variant == "50":
                    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
                elif model_variant == "101":
                    resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
                else:
                    assert False
            elif model_name == "resnext":
                if model_variant == "50":
                    resnet = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
                else:
                    assert False
            else:
                assert False
        else:
            assert False
        model = ImageCaption(vocab_size=90, resnet=resnet)
    elif args.model.startswith("imagecaptionwithbox") or args.model.startswith("icwb"):
        embed_parent = 0
        if args.model[-1] == "2":
            embed_parent = 1
        elif args.model[-1] == "3":
            embed_parent = 2
        if args.model_resnet.startswith("vm://"):
            model_url = urlparse(args.model_resnet)
            model_name = model_url.netloc
            model_params = model_url.query.split("&")
            load_weight = ("load_weight" in model_params)
            copy_weight = ("copy_weight" in model_params)
            print(f"vm:// {model_name} ? load_weight={load_weight} & copy_weight={copy_weight}")
            from dt.models import VisModel
            resnet = VisModel(model=model_name, load_weight=load_weight, copy_weight=copy_weight).resnet
        else:
            from dt.trainer import Trainer as DtTrainer
            t = DtTrainer.load_checkpoint(args.model_resnet)
            resnet = t.get_inner_model().resnet
        model = ImageCaptionWithBox(resnet, vocab_size=90, embed_parent=embed_parent)
    elif args.model == 'detr':
        model = Detr(num_classes=90)
    else:
        t = Trainer.load_checkpoint(args.model)
        model = t.get_inner_model()
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

    if args.split_path is not None:
        split = np.load(args.split_path)
        split_train = split["train"]
        split_valid = split["valid"]
        split_test = split["test"]
    else:
        split_train, split_valid, split_test = None, None, None

    has_comma = (not args.no_comma)
    has_rect = (args.model.startswith("imagecaptionwithbox") or args.model.startswith("icwb") or args.model == 'detr')
    
    train_set = ImageCodeDataset(args.image_path, args.code_path, split_train, transform=PresetEval(),
                                 has_comma=has_comma, has_rect=has_rect)
    
    train_set.summary("> Train set")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                              num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        
    if split_valid is not None:
        valid_set = ImageCodeDataset(args.image_path, args.code_path, split_valid, transform=PresetEval(),
                                    has_comma=has_comma, has_rect=has_rect)
    
        valid_set.summary("> Valid set")
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    else:
        valid_loader = None
    
    trainer = Trainer(model=model, optimizer=optimizer, generator=generator,
                      is_ema=args.ema, use_amp=args.amp)
    trainer.epochs_early_stop = args.epochs_early_stop
    trainer.epochs_adjust_lr = args.epochs_adjust_lr
    trainer.early_stop = args.early_stop

    if args.grad_clip:
        trainer.grad_clip = 1.
        trainer.grad_clip_fn = nn.utils.clip_grad.clip_grad_value_

    if args.lr_find:
        trainer.lr_find(end_lr=100., step_mode='exp', epochs=100,
                        train_loader=train_loader, valid_loader=valid_loader)
        return
    
    if args.metric == "acc":
        metrics = SimpleMulticlassMetrics(90, scorer=MulticlassAccuracy)
    elif args.metric == "auc":
        metrics = SimpleMulticlassMetrics(90, scorer=MulticlassAUROC)
    else:
        assert False

    trainer.fit(epochs=args.epochs, train_loader=train_loader, valid_loader=valid_loader, 
                metrics=metrics, proof_of_concept=args.proof_of_concept)
    
    if split_test is not None:
        print("=" * 100)
        test_set = ImageCodeDataset(args.image_path, args.test_path, split_test, transform=PresetEval(),
                                    has_comma=has_comma, has_rect=has_rect)
    
        test_set.summary("> Test set")
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        
        trainer = Trainer.load_checkpoint("./BEST.pth.tar")
        _ = trainer.test(data_loader=test_loader, metrics=metrics, proof_of_concept=args.proof_of_concept)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)