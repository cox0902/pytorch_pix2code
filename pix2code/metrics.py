from typing import *

import time
import warnings 
from collections import deque
from datetime import timedelta
from tabulate import tabulate
import torch
from torcheval.metrics import *
from torcheval.metrics.metric import Metric
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion


class AverageMeter:
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.sum, self.count = 0, 0
        self.deque = deque(maxlen=self.window_size)

    def update(self, val, n: int = 1):
        self.deque.append(val)
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def smoothed_avg(self):
        return sum(self.deque) / len(self.deque)

    @property
    def val(self):
        return self.deque[-1]


def _tf(seconds) -> str:
    td = timedelta(seconds=seconds)
    mm, ss = divmod(td.seconds, 60)
    hh, mm = divmod(mm, 60)
    s = f'{hh}:{mm:02}:{ss:02}'
    if td.days:
        def plural(n):
            return n, abs(n) != 1 and "s" or ""
        s = ("%d day%s, " % plural(td.days)) + s
    return s
    

def _format_named_value(name: str, value, average = None, precision: int = 4):
    format_str = f"{{name}} {{value:.{precision}f}}"
    if average is not None:
        format_str += f" ({{average:.{precision}f}})"
        return format_str.format(name=name, value=value, average=average)
    return format_str.format(name=name, value=value)


def _format_named_meter(name: str, meter: AverageMeter, show_average: bool, precision: int = 4):
    if show_average:
        return _format_named_value(name, value=meter.val, average=meter.smoothed_avg, precision=precision)
    return _format_named_value(name, value=meter.val, precision=precision)


class Metrics:

    def __init__(self, metrics: Dict[str, Metric], scorer: Type[Metric] = None):
        self.start_time = time.perf_counter()
        self.batch_time = AverageMeter()
        self.batch_count = 0
        self.loss = AverageMeter(window_size=10)
        self.losses: Dict[str, AverageMeter] = {}
        self.metrics = [{
            'name': each_name,
            'meter': AverageMeter(),
            'scorer': each_scorer
        } for each_name, each_scorer in metrics.items()]
        self.scorer: Type[Metric] = scorer

        self.name_loss = "loss"
        self.name_predicts = "scores"
        self.name_targets = "targets"

    def reset(self, batch_count):
        self.batch_time.reset()
        self.start_time = time.perf_counter()
        self.batch_count = batch_count
        self.loss.reset()  # 
        for each in self.losses.values():
            each.reset()
        for metric in self.metrics:
            metric['scorer'].reset()
            metric['meter'].reset()

    def update(self, outputs: Dict[str, Any] = {}):
        #
        predicts = outputs[self.name_predicts] if self.name_predicts in outputs else None
        targets = outputs[self.name_targets] if self.name_targets in outputs else None
        if predicts is not None and targets is not None:
            for metric in self.metrics:
                metric['scorer'].update(predicts, targets)
                metric['meter'].update(metric['scorer'].compute())

        #
        loss = outputs[self.name_loss] if self.name_loss in outputs else None
        if loss is not None:
            if targets is not None:
                self.loss.update(loss, len(targets))
            else:
                self.loss.update(loss)

        for k, v in outputs.items():
            if k.startswith(self.name_loss + "/"):
                if k not in self.losses:
                    self.losses[k] = AverageMeter()
                self.losses[k].update(v)

        self.batch_time.update(time.perf_counter() - self.start_time)
        self.start_time = time.perf_counter()

    def compute(self, hypotheses, references) -> float:
        if self.scorer is None:
            return 0.0
        scorer: Metric = self.scorer()
        scorer.update(hypotheses, references)
        return scorer.compute()
    
    def format(self, show_scores: bool = True, show_average: bool = True, 
               show_batch_time: bool = True, show_loss: bool = True) -> str:
        agg_metrics = []
        if torch.cuda.is_available():
            GB = 1024.0 * 1024.0 * 1024.0
            ma, mr = torch.cuda.mem_get_info()
            # ma = torch.cuda.max_memory_allocated()
            # mr = torch.cuda.max_memory_reserved()
            agg_metrics.append(f"FREE {ma / GB:.2f} / {mr / GB:.2f} GB")
        if show_batch_time:
            str_inline = f"ETA {_tf(self.batch_time.sum)}"
            if self.batch_count > 0 and self.batch_count > self.batch_time.count:
                str_inline += f" / FIN {_tf(self.batch_time.avg * (self.batch_count - self.batch_time.count))}"
            agg_metrics.append(str_inline)
        if show_loss:
            str_inline = _format_named_meter("Loss", self.loss, show_average, precision=4)
            # str_inline = f"Loss {self.loss.val:.4f}"
            # if show_average:
            #     str_inline += f" ({self.loss.smoothed_avg:.4f})"
            agg_metrics.append(str_inline)

            i = 1
            for k, v in self.losses.items():
                if i % 5 == 0:
                    agg_metrics.append("\n")
                str_inline = _format_named_meter(k, v, show_average, precision=4)
                agg_metrics.append(str_inline)
                i += 1

        # if show_batch_time or show_loss:8
        #     agg_metrics.append("\n")
        if show_scores:
            for i, metric in enumerate(self.metrics):
                if i % 5 == 0:
                    agg_metrics.append("\n")
                str_inline = _format_named_meter(metric["name"], metric["meter"], show_average, precision=5)
                # str_inline = f'{metric["name"]} {metric["meter"].val:.5f}'
                # if show_average:
                #     str_inline += f' ({metric["meter"].avg:.5f})'
                agg_metrics.append(str_inline)
        return '\t'.join(agg_metrics)
    

class SimpleBinaryMetrics(Metrics):

    def __init__(self, metrics: Dict[str, Metric] = None, scorer: Type[Metric] = BinaryAUROC):
        if metrics is None:
            super().__init__({ 
                "Acc": BinaryAccuracy(),
                "AUC": BinaryAUROC() 
            }, scorer)
        else:
            super().__init__(metrics, scorer)

    def compute(self, hypotheses, references) -> float:
        score = super().compute(hypotheses, references)

        bcm = BinaryConfusionMatrix()
        bcm.update(hypotheses, references.long())
        m = bcm.compute().long()
        TP = m[0][0]
        FN = m[0][1]
        FP = m[1][0]
        TN = m[1][1]
        print(tabulate([["T", f"TP {m[0][0]}", f"FN {m[0][1]}"], ["F", f"FP {m[1][0]}", f"TN {m[1][1]}"]], 
                       headers=["", "P", "N"], tablefmt="psql"))
        print(f'* POS * Pre {TP / (TP + FP):.5f} Rec {TP / (TP + FN):.5f} F-1 {2 * TP / (2 * TP + FP + FN):.5f}')
        print(f'* NEG * Pre {TN / (TN + FN):.5f} Rec {TN / (TN + FP):.5f} F-1 {2 * TN / (2 * TN + FP + FN):.5f}')
        return score


class SimpleBinaryMetricsAcc(SimpleBinaryMetrics):

    def __init__(self):
        super().__init__(scorer=BinaryAccuracy)


class BinaryMetrics(SimpleBinaryMetrics):

    def __init__(self):
        super().__init__({ 
            "Acc": BinaryAccuracy(),
            "Pre": BinaryPrecision(), 
            "Rec": BinaryRecall(),
            "F-1": BinaryF1Score(),
            "AUC": BinaryAUROC() 
        })


class EmptyMetrics(Metrics):

    def __init__(self):
        super().__init__({})


class SimpleMulticlassMetrics(Metrics):

    def __init__(self, num_classes: int, metrics: Dict[str, Metric] = None, scorer: Type[Metric] = MulticlassAUROC):
        self.num_classes = num_classes
        if metrics is None:
            super().__init__({ 
                "Acc": MulticlassAccuracy(num_classes=num_classes),
                "AUC": MulticlassAUROC(num_classes=num_classes) 
            }, scorer)
        else:
            super().__init__(metrics, scorer)

    def compute(self, hypotheses, references) -> float:
        if self.scorer is None:
            return 0.0
        scorer: Metric = self.scorer(num_classes=self.num_classes)
        scorer.update(hypotheses, references)
        return scorer.compute()
    

# class MulticlassMetrics(SimpleMulticlassMetrics):

#     def __init__(self, num_classes: int, scorer: Type[Metric] = MulticlassAUROC):
#         super().__init__(num_classes, {
#             "Acc": MulticlassAccuracy(num_classes=num_classes),
#             "Pre": MulticlassPrecision(num_classes=num_classes), 
#             "Rec": MulticlassRecall(num_classes=num_classes),
#             "F-1": MulticlassF1Score(num_classes=num_classes),
#             "AUC": MulticlassAUROC(num_classes=num_classes) 
#         }, scorer)


class SimpleLossMetrics(Metrics):

    def __init__(self):
        super().__init__({})

    def compute(self, hypotheses, references):
        return -self.loss
    

# --- v2.0

class Scorer:

    def __init__(self):
        self.scorer = None

    def reset(self):
        self.scorer.reset()

    def update(self, outputs):
        self.scorer.update(outputs)

    def compute(self):
        return self.scorer.compute()
    
    def format(self):
        return [_format_named_value(self.name, self.compute(), precision=5)]


class CompoundScorer(Scorer):

    def __init__(self):
        self.scorers = []

    def reset(self):
        for each in self.scorers:
            each.reset()
    
    def update(self, outputs):
        for each in self.scorers:
            each.update(outputs)

    def compute(self):
        sum_score = 0
        for each in self.scorers:
            sum_score += each.compute()
        return sum_score
    
    def format(self):
        return [each.format() for each in self.scorers]


registered_scores = {}


class SimpleMetricScorer(Scorer):

    def __init__(self, name, metric, name_hyp, name_ref):
        super().__init__()
        self.name = name
        self.scorer = metric
        self.name_hyp = name_hyp
        self.name_ref = name_ref

    def update(self, outputs):
        self.scorer.update(outputs[self.name_hyp], outputs[self.name_ref])


registered_scores["acc"] = SimpleMetricScorer("acc", MulticlassAccuracy(num_classes=90), "scores", "targets")
registered_scores["auc"] = SimpleMetricScorer("auc", MulticlassAUROC(num_classes=90), "scores", "targets")


class MapScorer(Scorer):

    def __init__(self):
        super().__init__()
        self.name = "map"
        self.scorer = MeanAveragePrecision()

    def update(self, outputs):
        in_preds = [{
            "boxes": outputs["preds_box"],
            "scores": outputs["scores_lbl"],
            "labels": outputs["preds_lbl"]
        }]
        in_target = [{
            "boxes": outputs["truth_box"],
            "labels": outputs["truth_lbl"]
        }]
        self.scorer.update(in_preds, in_target)

    def compute(self):
        return self.scorer.compute()["map"]


# registered_scores["map"] = MapScorer()


class IouScorer(Scorer):

    def __init__(self, name_hyp, name_ref):
        super().__init__()
        self.name = "iou"
        self.name_hyp = name_hyp
        self.name_ref = name_ref
        self.scorer = IntersectionOverUnion()
        self.ious = []

    @staticmethod
    def _compute_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x1, y1, x2, y2 = boxes1.unbind(dim=-1)
        x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

        # Intersection keypoints
        xkis1 = torch.max(x1, x1g)
        ykis1 = torch.max(y1, y1g)
        xkis2 = torch.min(x2, x2g)
        ykis2 = torch.min(y2, y2g)

        intsctk = torch.zeros_like(x1)
        mask = (ykis2 > ykis1) & (xkis2 > xkis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

        # return intsctk, unionk
        return intsctk / (unionk + eps)
    
    def update(self, outputs):
        iou = IouScorer._compute_iou(outputs[self.name_hyp], outputs[self.name_ref])
        # print(iou.shape)
        self.ious.append(iou)

    def compute(self):
        mean_iou = torch.cat(self.ious).mean()
        # print(mean_iou.shape)
        # self.ious = []
        return mean_iou.item()

registered_scores["iou"] = IouScorer("scores", "targets")


def _handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


def _iou_score(tp, fp, fn, tn):
    return tp / (tp + fp + fn)


def _get_stats_multilabel(
    output: torch.LongTensor, target: torch.LongTensor
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    batch_size, num_classes, *dims = target.shape
    output = output.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


def _compute_metric(
    metric_fn,
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division="warn",
    **metric_kwargs,
) -> float:
    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(
            f"Class weights should be provided for `{reduction}` reduction"
        )

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = torch.tensor(class_weights).to(tp.device)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    elif reduction == "macro":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score * class_weights).mean()

    elif reduction == "weighted":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score * class_weights).sum()

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()

    elif reduction == "macro-imagewise" or reduction == "weighted-imagewise":
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score.mean(0) * class_weights).mean()

    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise,"
            + "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


class MaskIouScorer(Scorer):

    def __init__(self):
        self.name = "mis"
        self.reset()

    def reset(self):
        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []

    def update(self, outputs):
        prob_mask = outputs["preds_box"].sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = _get_stats_multilabel(pred_mask.long(), outputs["truth_box"].long())
        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.tn.append(tn)
    
    def compute_iou(self, reduction):
        tp = torch.cat(self.tp)
        fp = torch.cat(self.fp)
        fn = torch.cat(self.fn)
        tn = torch.cat(self.tn)
        return _compute_metric(_iou_score, tp=tp, fp=fp, fn=fn, tn=tn, reduction=reduction)

    def compute(self):
        return self.compute_iou("micro")
    
    def format(self):
        return [
            _format_named_value("iou", self.compute_iou("micro"), precision=5),
        ]
    

registered_scores["mis"] = MaskIouScorer()
    

class MaskIouCompoundScorer(Scorer):

    def __init__(self):
        self.name = "mics"
        self.reset()

    def reset(self):
        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []

    def update(self, outputs):
        prob_mask = outputs["preds_box"].sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = _get_stats_multilabel(pred_mask.long(), outputs["truth_box"].long())
        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.tn.append(tn)
    
    def compute_iou(self, reduction):
        tp = torch.cat(self.tp)
        fp = torch.cat(self.fp)
        fn = torch.cat(self.fn)
        tn = torch.cat(self.tn)
        return _compute_metric(_iou_score, tp=tp, fp=fp, fn=fn, tn=tn, reduction=reduction)

    def compute(self):
        return self.compute_iou("micro-imagewise") + self.compute_iou("micro")
    
    def format(self):
        return [
            _format_named_value("iou_img", self.compute_iou("micro-imagewise"), precision=5),
            _format_named_value("iou_all", self.compute_iou("micro"), precision=5)
        ]

    
registered_scores["mics"] = MaskIouCompoundScorer()


# ---

class BleuScorer(Scorer):

    def __init__(self, n: int = 4):
        super().__init__()
        self.name = f"bleu{n}"
        self.scorer = BLEUScore(n_gram=n)

    def update(self, outputs):
        formated_candidates = []
        formated_references = []
        for each_source, each_target in zip(outputs["sources"], outputs["targets"]):
            print(each_source)
            print(each_target)
            formated_candidates.append(" ".join([str(id) for id in each_source]))
            formated_references.append([" ".join([str(id) for id in each_target])])
        self.scorer.update(formated_candidates, formated_references)


registered_scores["bleu1"] = BleuScorer(1)
registered_scores["bleu2"] = BleuScorer(2)
registered_scores["bleu3"] = BleuScorer(3)
registered_scores["bleu4"] = BleuScorer(4)


# ---

class AdvMetrics:

    def __init__(self, metrics: List[Scorer] = [], reduction = "sum"):
        self.start_time = time.perf_counter()
        self.batch_time = AverageMeter()
        self.batch_count = 0
        self.metrics = metrics
        self.reduction = reduction if reduction is not None else "sum"

    def add_metric(self, name):
        self.metrics.append(registered_scores[name])

    def add_metric_obj(self, obj):
        self.metrics.append(obj)

    def reset(self, batch_count):
        self.batch_time.reset()
        self.start_time = time.perf_counter()
        self.batch_count = batch_count
        for metric in self.metrics:
            metric.reset()

    def update(self, outputs: Dict[str, Any] = {}):
        #
        for metric in self.metrics:
            metric.update(outputs)

        self.batch_time.update(time.perf_counter() - self.start_time)
        self.start_time = time.perf_counter()

    def compute(self) -> float:
        if self.reduction == "sum":
            agg = 0
            for metric in self.metrics:
                agg += metric.compute()
            return agg
        elif self.reduction == "avg":
            agg = 0
            for metric in self.metrics:
                agg += metric.compute()
            return agg / len(self.metrics)
        else:
            for metric in self.metrics:
                if metric.name == self.reduction:
                    return metric.compute()
            assert False

    def format(self, show_scores: bool = True, show_batch_time: bool = True) -> str:
        agg_metrics = []

        if torch.cuda.is_available():
            MB = 1024.0 * 1024.0
            ma, mr = torch.cuda.mem_get_info()
            # ma = torch.cuda.max_memory_allocated()
            # mr = torch.cuda.max_memory_reserved()
            agg_metrics.append(f"{int(ma / MB)} MB / {int(mr / MB)} MB")

        if show_batch_time:
            str_inline = f"ETA {_tf(self.batch_time.sum)}"
            if self.batch_count > 0 and self.batch_count > self.batch_time.count:
                str_inline += f" / FIN {_tf(self.batch_time.avg * (self.batch_count - self.batch_time.count))}"
            agg_metrics.append(str_inline)

        # if show_batch_time or show_loss:8
        #     agg_metrics.append("\n")
        if show_scores:
            for i, metric in enumerate(self.metrics):
                if i % 5 == 0:
                    agg_metrics.append("\n")
                # str_inline = _format_named_value(metric.name, metric.compute(), precision=5)
                # str_inline = f'{metric["name"]} {metric["meter"].val:.5f}'
                # if show_average:
                #     str_inline += f' ({metric["meter"].avg:.5f})'
                agg_metrics.extend(metric.format())
        return '\t'.join(agg_metrics)
    