from typing import *

import argparse
from pathlib import Path

import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC

from pix2code.utils import seed_everything
from pix2code.trainer import Trainer
from pix2code.metrics import SimpleMulticlassMetrics
from pix2code.dataset import ImageCodeDataset
from pix2code.transforms import PresetEval
from pix2code.model import Pix2Code


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--lr-find", action="store_true")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--opt", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--metric", default="auc", type=str)

    parser.add_argument("--image-path", type=str)
    parser.add_argument("--split-path", type=str)
    parser.add_argument("--code-path", type=str)
    parser.add_argument("--ignore-path", type=str)

    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    
    parser.add_argument("--grad-clip", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", default=3600, type=int)

    parser.add_argument("--no-comma", action="store_false")

    return parser


def main(args):
    print(args)

    generator, seed_worker = seed_everything(args.seed)

    model = Pix2Code(vocab_size=90)
    criterion = nn.CrossEntropyLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    split = np.load(args.split_path)
    split_train = split["train"]
    split_valid = split["valid"]
    split_test = split["test"]

    if args.ignore_path is not None:
        ignores = np.load(args.ignore_path)
        split_train = np.array([each for each in split_train if each not in ignores])
        split_valid = np.array([each for each in split_train if each not in ignores])
        split_test = np.array([each for each in split_train if each not in ignores])
    
    if not args.no_comma:
        train_set = ImageCodeDataset(args.image_path, args.code_path, split_train, transform=PresetEval())
    else:
        train_set = ImageCodeDataset(args.image_path, args.code_path, split_train, transform=PresetEval(),
                                     has_comma=False)
    
    train_set.summary()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                              num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        
    if not args.no_comma:
        valid_set = ImageCodeDataset(args.image_path, args.code_path, split_valid, transform=PresetEval())
    else:
        valid_set = ImageCodeDataset(args.image_path, args.code_path, split_valid, transform=PresetEval(),
                                     has_comma=False)
   
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, generator=generator,
                      is_ema=args.ema, use_amp=args.amp)
    trainer.epochs_early_stop = 20

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
    
    print("=" * 100)
    if not args.no_comma:
        test_set = ImageCodeDataset(args.image_path, args.code_path, split_test, transform=PresetEval())
    else:
        test_set = ImageCodeDataset(args.image_path, args.code_path, split_test, transform=PresetEval(),
                                     has_comma=False)
   
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
                         
    trainer = Trainer.load_checkpoint("./BEST.pth.tar")
    _ = trainer.test(data_loader=test_loader, metrics=metrics, proof_of_concept=args.proof_of_concept)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)