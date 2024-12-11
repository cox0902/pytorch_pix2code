
from typing import *

import hashlib
import numpy as np
import torch
from torch import nn
from torch import optim
# from torch.cuda.amp import autocast, GradScaler  # torch 2.4.0 warning
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import LRScheduler
import torch.utils
from torch.utils.data import DataLoader

from .metrics import Metrics, EmptyMetrics
from .utils import get_rng_state


class ExponentialMovingAverage(AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    decay: float = 0.999

    def __init__(self, model):
        super().__init__(model, multi_avg_fn=self.ema_update, use_buffers=True)

    @staticmethod
    def ema_update(ema_param_list, current_param_list, _):
        with torch.no_grad():
            # foreach lerp only handles float and complex
            if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(ema_param_list[0]):
                torch._foreach_lerp_(ema_param_list, current_param_list, 1 - ExponentialMovingAverage.decay)
            else:
                for p_ema, p_model in zip(ema_param_list, current_param_list):
                    p_ema.copy_(p_ema * ExponentialMovingAverage.decay + p_model * (1 - ExponentialMovingAverage.decay))


class LinearLR(LRScheduler):

    def __init__(self, optimizer: optim.Optimizer, end_lr: float, epochs: int):
        self.end_lr = end_lr
        self.epochs = epochs
        super().__init__(optimizer)

    def get_lr(self):
        r = self.last_epoch / (self.epochs - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]
    

class ExponentialLR(LRScheduler):

    def __init__(self, optimizer: optim.Optimizer, end_lr: float, epochs: int):
        self.end_lr = end_lr
        self.epochs = epochs
        super().__init__(optimizer)

    def get_lr(self):
        r = self.last_epoch / (self.epochs - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
    


class Trainer:

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, generator: torch.Generator,
                 is_ema: bool = False, use_amp: bool = False):
        self.print_freq: int = 100
        self.early_stop: bool = True
        
        self.warmup_epochs: int = 5
        self.epochs_early_stop: int = 10
        self.epochs_adjust_lr: int = 4
        self.grad_clip: Optional[float] = None  # 5.
        self.grad_clip_fn = nn.utils.clip_grad.clip_grad_norm_

        self.epoch: int = 0
        self.epochs_since_improvement: int = 0
        self.best_score: float = 0.

        self.seed = None
        self.generator = generator
        self.state = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Trainable parameters: {params:,}")

        if model is not None:
            self.model = model.to(self.device)

        self.ema_model = None
        if is_ema:
            self.ema_model = ExponentialMovingAverage(self.model)

        self.scaler = GradScaler("cuda") if use_amp else None

        self.optimizer = optimizer

    def adjust_learning_rate(self, shrink_factor: float):
        print("\nDECAYING learning rate...")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * shrink_factor
        print(f"- The new learning rate is {self.optimizer.param_groups[0]['lr']}\n")

    def save_checkpoint(self, epoch: int, epochs_since_improvement: int, score, is_best: bool,
                        save_checkpoint: bool = True):
        state = {
            'seed': torch.initial_seed(),
            'epoch': epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'score': score,
            'model': self.model,
            'optimizer': self.optimizer,
            'scaler': self.scaler,
            'state': get_rng_state(self.generator)
        }
        if self.ema_model is not None:
            state['ema_model'] = self.ema_model
        if is_best:
            torch.save(state, 'BEST.pth.tar')
        elif save_checkpoint:
            torch.save(state, 'checkpoint.pth.tar')

    @staticmethod
    def load_checkpoint(save_file: str = None, is_best: bool = True) -> "Trainer":
        if save_file is None:
            if is_best:
                save_file = 'BEST.pth.tar'
            else:
                save_file = 'checkpoint.pth.tar'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(save_file, map_location=device, weights_only=True)  # torch 2.4.0 warning

        model: nn.Module = saved['model']
        is_ema = 'ema_model' in saved
        optimizer = saved['optimizer']
        scaler = saved['scaler'] if 'scaler' in saved else None
 
        md5 = hashlib.md5()
        for arg in model.parameters():
            x = arg.data
            if hasattr(x, "cpu"):
                md5.update(x.cpu().numpy().data.tobytes())
            elif hasattr(x, "numpy"):
                md5.update(x.numpy().data.tobytes())
            elif hasattr(x, "data"):
                md5.update(x.data.tobytes())
            else:
                try:
                    md5.update(x.encode("utf-8"))
                except:
                    md5.update(str(x).encode("utf-8"))

        print(f"Loaded {md5.hexdigest()}")
        print(f"  from '{save_file}'")
        if 'seed' in saved:
            print(f"- seed      : {saved['seed']}")
        if is_ema:
            print(f"- is_ema    : True")
        if scaler is not None:
            print(f"- is_amp    : True")
        print(f"- epoch     : {saved['epoch']}")
        print(f"- epochs_since_improvement: {saved['epochs_since_improvement']}")
        print(f"- score     : {saved['score']}")

        trainer = Trainer(model=model, optimizer=optimizer, generator=None)
        trainer.seed = saved["seed"] if "seed" in saved else None
        trainer.epoch = saved["epoch"] + 1
        trainer.epochs_since_improvement = saved["epochs_since_improvement"]
        trainer.best_score = saved["score"]
        trainer.scaler = scaler
        trainer.state = saved['state'] if 'state' in saved else None
        if is_ema:
            trainer.ema_model = saved['ema_model']
        return trainer
    
    def to_device(self, data: Dict) -> Dict:
        for k, v in data.items():
            data[k] = v.to(self.device)
        return data

    def get_model(self) -> nn.Module:
        if self.ema_model is not None:
            return self.ema_model
        return self.model
    
    def get_inner_model(self) -> nn.Module:
        if self.ema_model is not None:
            return self.ema_model.module
        return self.model

    def train(self, data_loader: DataLoader, metrics: Metrics, epoch: int, proof_of_concept: bool = False) -> float:
        self.model.train()
        metrics.reset(len(data_loader))

        # total_loss = 0

        print()
        for i, batch in enumerate(data_loader):
            batch = self.to_device(batch)
            # targets = batch["target"]

            with autocast("cuda", enabled=self.scaler is not None):
                outputs = self.model(batch)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(outputs["loss"]).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    self.grad_clip_fn(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs["loss"].backward()

                if self.grad_clip is not None:
                    self.grad_clip_fn(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
            
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
                if epoch < self.warmup_epochs:
                    self.ema_model.n_averaged.fill_(0)

            metrics.update(outputs)
            # total_loss += outputs["loss"].item()

            if i % self.print_freq == 0:
                print(f"Epoch [{epoch}][{i + 1}/{len(data_loader)}]\t{metrics.format()}")
                
            if proof_of_concept:
                break

        print(f"Epoch [{epoch}][{i + 1}/{len(data_loader)}]\t{metrics.format()}")
        # return total_loss
        return metrics.loss.sum
        
    def valid(self, data_loader: DataLoader, metrics: Metrics, proof_of_concept: bool = False) -> Tuple[float, float]:
        model = self.get_model()
        model.eval()        
        metrics.reset(len(data_loader))

        references = []
        hypotheses = []
        total_loss = 0

        print()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch = self.to_device(batch)
                # targets = batch["target"]
                outputs = model(batch)

                metrics.update(outputs)
                # total_loss += outputs["loss"].item() * len(outputs["targets"])

                if i % self.print_freq == 0:
                    print(f'Validation [{i + 1}/{len(data_loader)}]\t{metrics.format()}')

                references.extend(outputs["targets"])
                hypotheses.extend(outputs["scores"])

                if proof_of_concept:
                    break

            print(f'Validation [{i + 1}/{len(data_loader)}]\t{metrics.format()}')
            total_loss = metrics.loss.avg

            hypotheses = torch.stack(hypotheses)
            references = torch.stack(references)
            metrics.reset(len(data_loader))
            metrics.update({ "scores": hypotheses, "targets": references })
            print(f'\n* {metrics.format(show_average=False, show_batch_time=False, show_loss=False)}')

        return metrics.compute(hypotheses, references), total_loss
    
    def test(self, data_loader: DataLoader, metrics: Metrics, hook: Optional[str] = None, 
             proof_of_concept: bool = False):
        model = self.get_model()
        model.eval()
        metrics.reset(len(data_loader))

        references = []
        hypotheses = []
        activations = []

        print()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch = self.to_device(batch)
                # targets = batch["target"]

                if hook is not None:
                    activation = {}
                    def fn_hook(model, input, output):
                        activation[hook] = output.detach()

                    if self.ema_model is not None:
                        handler = getattr(model.module.resnet, hook).register_forward_hook(fn_hook)
                    else:
                        handler = getattr(model.resnet, hook).register_forward_hook(fn_hook)

                    outputs = model(batch)
                    handler.remove()

                    # print(activation[hook].shape)
                    activations.extend(activation[hook].squeeze().cpu().numpy())  # !! TODO: add n to squeeze
                else:
                    outputs = model(batch)
               
                metrics.update()  # 

                if i % self.print_freq == 0:
                    print(f'Test [{i + 1}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')

                references.extend(outputs["targets"])
                hypotheses.extend(outputs["scores"])

                if proof_of_concept:
                    break

            print(f'Test [{i + 1}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')
            
            hypotheses = torch.stack(hypotheses)
            references = torch.stack(references)
            metrics.reset(len(data_loader))
            metrics.update({ "scores": hypotheses, "targets": references })
            print(f'\n* {metrics.format(show_average=False, show_batch_time=False, show_loss=False)}')

            metrics.compute(hypotheses, references)

        if hook is None:
            return hypotheses, references, None
        return hypotheses, references, np.array(activations)

    def lr_find(self, end_lr: float, step_mode: Literal["exp", "linear"], epochs: int, 
                train_loader: DataLoader, valid_loader: Optional[DataLoader], 
                smooth_factor: float = 0.05, diverge_threshold: float = 5.):
        history_lr, history_loss = [], []
        metrics = EmptyMetrics()

        print("\n= Warmup")
        self.train(train_loader, metrics, 0, proof_of_concept=True)
        # self.save_checkpoint(epoch=0, epochs_since_improvement=0, score=loss, is_best=False)

        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, epochs)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, epochs)
        else:
            assert False

        best_loss = None
        for epoch in range(epochs):
            loss = self.train(train_loader, metrics, epoch + 1, proof_of_concept=True)

            if not all([torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None]):
                print("= Early stop while grad nan")
                break

            if valid_loader is not None:
                _, loss = self.valid(valid_loader, metrics)

            history_lr.append(lr_schedule.get_lr()[0])
            lr_schedule.step()

            if best_loss is None:
                best_loss = loss
            else:
                loss = smooth_factor * loss + (1 - smooth_factor) * history_loss[-1]
                if loss < best_loss:
                    best_loss = loss
            
            history_loss.append(loss)
            if loss >= diverge_threshold * best_loss:
                print("= Early stop while diverge")
                break

        np.savez("lr_find.npz", lr=history_lr, loss=history_loss)

    def fit(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader, metrics: Metrics,
            save_checkpoint: bool = True, proof_of_concept: bool = False):
        assert self.generator is not None

        epochs_since_improvement: int = self.epochs_since_improvement
        best_score: float = self.best_score

        for epoch in range(self.epoch, epochs):
            if epoch >= self.warmup_epochs:
                if epochs_since_improvement == self.epochs_early_stop:
                    if self.early_stop:
                        break
                if epochs_since_improvement > 0 and epochs_since_improvement % self.epochs_adjust_lr == 0:
                    self.adjust_learning_rate(0.8)

            self.train(data_loader=train_loader, metrics=metrics, epoch=epoch, proof_of_concept=proof_of_concept)

            recent_score, _ = self.valid(data_loader=valid_loader, metrics=metrics, proof_of_concept=proof_of_concept)
            
            if epoch >= self.warmup_epochs or proof_of_concept:
                is_best = recent_score > best_score
                best_score = max(recent_score, best_score)

                if not is_best:
                    epochs_since_improvement += 1
                    print(f"\nEpochs since last improvement: {epochs_since_improvement} ({best_score})\n")  # [OK]
                else:
                    epochs_since_improvement = 0

                # save checkpoint
                self.save_checkpoint(epoch=epoch, epochs_since_improvement=epochs_since_improvement, 
                                    score=recent_score, is_best=is_best, 
                                    save_checkpoint=save_checkpoint)
            
            if proof_of_concept:
                break

    def predict(self, data_loader: DataLoader, proof_of_concept: bool = False):
        model = self.get_model()
        model.eval()

        metrics = EmptyMetrics()
        metrics.reset(len(data_loader))

        predicts_collected = {}

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch = self.to_device(batch)

                if self.ema_model is not None:
                    predicts = model.module.predict(batch)
                else:
                    predicts = model.predict(batch)

                for k, v in predicts.items():
                    if k not in predicts_collected:
                        predicts_collected[k] = []
                    predicts_collected[k].extend(v.cpu())

                metrics.update(None, None)  # 

                if i % self.print_freq == 0:
                    print(f'Predict [{i + 1}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')

                if proof_of_concept:
                    break

        print(f'Predict [{i + 1}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')
        return predicts_collected
        