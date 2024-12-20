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
from pix2code.models import Pix2Code, ImageCaption, ImageCaptionWithBox, Detr


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--model", type=str)
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

    parser.add_argument("--no-comma", action="store_false")

    return parser


def main(args):
    print(args)

    generator, seed_worker = seed_everything(args.seed)

    if args.model == "pix2code":
        model = Pix2Code(vocab_size=90)
    elif args.model == 'imagecaption':
        model = ImageCaption(vocab_size=90)
    elif args.model in ['imagecaptionwithbox', 'icwb']:
        model = ImageCaptionWithBox(vocab_size=90)
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

    split = np.load(args.split_path)
    split_train = split["train"]
    split_valid = split["valid"]
    split_test = split["test"]

    has_comma = (not args.no_comma)
    has_rect = (args.model in ['imagecaptionwithbox', 'icwb', 'detr'])
    
    train_set = ImageCodeDataset(args.image_path, args.code_path, split_train, transform=PresetEval(),
                                 has_comma=has_comma, has_rect=has_rect)
    
    train_set.summary("> Train set")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                              num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        
    valid_set = ImageCodeDataset(args.image_path, args.code_path, split_valid, transform=PresetEval(),
                                 has_comma=has_comma, has_rect=has_rect)
   
    valid_set.summary("> Valid set")
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
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