from typing import *

import argparse
from pathlib import Path

import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from pix2code.utils import seed_everything
from pix2code.trainer import Trainer
from pix2code.metrics import SimpleMulticlassMetrics
from pix2code.dataset import ImageCodeDataset
from pix2code.transforms import PresetEval
from pix2code.model import Pix2Code


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--proof-of-concept", action="store_true")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--opt", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--image-path", type=str)
    parser.add_argument("--split-path", type=str)
    parser.add_argument("--code-path", type=str)

    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", default=120, type=int)

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
    train_set = ImageCodeDataset(args.image_path, args.code_path, split["train"], transform=PresetEval())
    train_set.summary()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                              num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        
    valid_set = ImageCodeDataset(args.image_path, args.code_path, split["valid"], transform=PresetEval())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, generator=generator,
                      is_ema=args.ema, use_amp=args.amp)
    trainer.fit(epochs=args.epochs, train_loader=train_loader, valid_loader=valid_loader, 
                metrics=SimpleMulticlassMetrics(90), proof_of_concept=args.proof_of_concept)
    
    print("=" * 100)
    # test_set = ImageCodeDataset(image_path / "images.hdf5", args.code_path, split["test"], 
    #                             transform=PresetEval())
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
                         
    trainer = Trainer.load_checkpoint("./BEST.pth.tar")
    # _ = trainer.test(data_loader=test_loader, metrics=SimpleBinaryMetrics(), proof_of_concept=args.proof_of_concept)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)