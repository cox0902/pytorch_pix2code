from typing import *

from functools import partial

import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torcheval.metrics import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC

from pix2code.utils import seed_everything
from pix2code.utils.config import get_args_parser, check_model, parse_model
from pix2code.trainer import Trainer
from pix2code.metrics import SimpleMulticlassMetrics, SimpleLossMetrics, AdvMetrics, SimpleMetricScorer
from pix2code.dataset import ImageCodeDataset
from pix2code.transforms import PresetEval
from pix2code.models import get_model_class_by_name
from pix2code.generators import GreedySearch, BeamSearch


def build_model(args, data_set):
    model_name, model_params = parse_model(args.model)
    assert model_name is not None

    model_class = get_model_class_by_name(model_name)
    assert model_class is not None

    if args.proof_of_concept:
        model_params["proof_of_concept"] = True

    model_backbone = args.backbone
    assert model_backbone is not None
    t = Trainer.load_checkpoint(model_backbone)
    model_params["backbone"] = t.get_inner_model()
    
    model_params["vocab_size"] = 90
    model_params["max_len"] = data_set.max_len

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


def create_metrics():
    metrics = AdvMetrics(reduction="sum")
    # metrics.add_metric_obj(SimpleMetricScorer("delete/acc", BinaryAccuracy(), "predict/delete", "targets/delete"))
    # metrics.add_metric_obj(SimpleMetricScorer("delete/auc", BinaryAUROC(), "predict/delete", "targets/delete"))
    metrics.add_metric_obj(SimpleMetricScorer("delete/acc", MulticlassAccuracy(num_classes=2), "predict/delete", "targets/delete"))
    metrics.add_metric_obj(SimpleMetricScorer("delete/auc", MulticlassAUROC(num_classes=2), "predict/delete", "targets/delete"))
    metrics.add_metric_obj(SimpleMetricScorer("update/acc", MulticlassAccuracy(num_classes=90), "predict/update", "targets/update"))
    metrics.add_metric_obj(SimpleMetricScorer("update/auc", MulticlassAUROC(num_classes=90), "predict/update", "targets/update"))
    # metrics.add_metric_obj(SimpleMetricScorer("insdel/acc", BinaryAccuracy(), "predict/insdel", "targets/insdel"))
    # metrics.add_metric_obj(SimpleMetricScorer("insdel/auc", BinaryAUROC(), "predict/insdel", "targets/insdel"))
    metrics.add_metric_obj(SimpleMetricScorer("insdel/acc", MulticlassAccuracy(num_classes=2), "predict/insdel", "targets/insdel"))
    metrics.add_metric_obj(SimpleMetricScorer("insdel/auc", MulticlassAUROC(num_classes=2), "predict/insdel", "targets/insdel"))
    metrics.add_metric_obj(SimpleMetricScorer("insupd/acc", MulticlassAccuracy(num_classes=90), "predict/insupd", "targets/insupd"))
    metrics.add_metric_obj(SimpleMetricScorer("insupd/auc", MulticlassAUROC(num_classes=90), "predict/insupd", "targets/insupd"))
    return metrics


def main(args):
    print(args)

    generator, seed_worker = seed_everything(args.seed)

    #

    has_rect, norm_rect, mask_rect = check_model(args.model)

    if has_rect and not ImageCodeDataset.is_support_has_rect(args.image_path, args.code_path):
        print("!! Warning: has_rect is not supported by dataset!")
        has_rect = False
        norm_rect = False

    if args.split_path is not None:
        split = np.load(args.split_path)
        split_train = split["train"]
        split_valid = split["valid"]
        split_test = split["test"]
    else:
        split_train, split_valid, split_test = None, None, None

    has_comma = (not args.no_comma)
    
    if not args.test_only:
        train_set = ImageCodeDataset(args.image_path, 
                                    args.code_path, 
                                    split_train, 
                                    transform=PresetEval(),
                                    multi_label=args.multi_label, 
                                    label_aug_prob=args.label_aug_prob,
                                    has_comma=has_comma, 
                                    has_rect=has_rect, 
                                    mask_rect=mask_rect,
                                    has_tree=args.has_tree,
                                    mask_tree=args.mask_tree,
                                    force_add_channel=args.force_add_channel)
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
                                        multi_label=args.multi_label,
                                        has_comma=has_comma, 
                                        has_rect=has_rect, 
                                        mask_rect=mask_rect,
                                        has_tree=args.has_tree,
                                        mask_tree=args.mask_tree,
                                        force_add_channel=args.force_add_channel)
            valid_set.normalize_rect = norm_rect
            valid_set.summary("> Valid set")
            valid_loader = DataLoader(valid_set, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    pin_memory=args.pin_memory)
        elif args.force_valid:
            valid_loader = train_loader
        else:
            valid_loader = None

    if args.resume is not None:

        trainer = Trainer.load_checkpoint(args.resume)
        assert trainer.seed == args.seed
        generator, seed_worker = seed_everything(args.seed, trainer.state)
        trainer.generator = generator

    else:

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

    #

    if args.lr_find:
        trainer.lr_find(end_lr=100., 
                        step_mode='exp', 
                        epochs=100,
                        train_loader=train_loader, 
                        valid_loader=valid_loader)
        return
    
    # if args.metric == "acc":
    #     metrics = SimpleMulticlassMetrics(90, scorer=MulticlassAccuracy)
    # elif args.metric == "auc":
    #     metrics = SimpleMulticlassMetrics(90, scorer=MulticlassAUROC)
    # # elif args.metric == "acc+":
    # #     metrics = MulticlassMetrics(90, scorer=MulticlassAccuracy)
    # # elif args.metric == "auc+":
    # #     metrics = MulticlassMetrics(90, scorer=MulticlassAUROC)
    # elif args.metric == "loss":
    #     metrics = SimpleLossMetrics()
    # else:
    #     assert False

    # if args.eval_metric is None:
    #     eval_metrics = metrics
    # else:
    #     eval_metrics = AdvMetrics(reduction=args.stop_metric)
    #     ems = args.eval_metric.split("+")
    #     for each in ems:
    #         eval_metrics.add_metric(each)

    metrics = SimpleLossMetrics()
    eval_metrics = create_metrics()

    if not args.test_only:
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
                                    multi_label=args.multi_label,
                                    has_comma=has_comma, 
                                    has_rect=has_rect, 
                                    mask_rect=mask_rect,
                                    has_tree=args.has_tree,
                                    mask_tree=args.mask_tree,
                                    force_add_channel=args.force_add_channel)
        test_set.normalize_rect = norm_rect
        test_set.summary("> Test set")
        test_loader = DataLoader(test_set, 
                                 batch_size=args.batch_size, 
                                 shuffle=False, 
                                 pin_memory=args.pin_memory)
        
        if not args.test_only:
            trainer = Trainer.load_checkpoint("./BEST.pth.tar")
        _ = trainer.test(data_loader=test_loader, 
                         metrics=eval_metrics, 
                         proof_of_concept=args.proof_of_concept)


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--backbone", type=str)

    args = parser.parse_args()
    main(args)