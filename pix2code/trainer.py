
from typing import Dict, Optional

import hashlib
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel
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


class Trainer:

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, generator: torch.Generator,
                 is_ema: bool = False, use_amp: bool = False):
        self.print_freq: int = 100
        
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

        self.scaler = GradScaler() if use_amp else None

        if criterion is not None:
            self.criterion = criterion.to(self.device)
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
            'criterion': self.criterion,
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
        saved = torch.load(save_file, map_location=device)

        model: nn.Module = saved['model']
        is_ema = 'ema_model' in saved
        criterion = saved['criterion'] if 'criterion' in saved else None
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

        trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, generator=None)
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

    def train(self, data_loader: DataLoader, metrics: Metrics, epoch: int, proof_of_concept: bool = False):
        self.model.train()
        metrics.reset(len(data_loader))

        print()
        for i, batch in enumerate(data_loader):
            batch = self.to_device(batch)
            # targets = batch["target"]

            with autocast(enabled=self.scaler is not None):
                logits, predicts, targets = self.model(batch)
                loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    self.grad_clip_fn(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.grad_clip is not None:
                    self.grad_clip_fn(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
            
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
                if epoch < self.warmup_epochs:
                    self.ema_model.n_averaged.fill_(0)

            metrics.update(predicts=predicts.squeeze(), targets=targets.squeeze(), loss=loss.item())

            if i % self.print_freq == 0:
                print(f"Epoch [{epoch}][{i + 1}/{len(data_loader)}]\t{metrics.format()}")
                
            if proof_of_concept:
                break

        print(f"Epoch [{epoch}][{i + 1}/{len(data_loader)}]\t{metrics.format()}")
        
    def valid(self, data_loader: DataLoader, metrics: Metrics, proof_of_concept: bool = False) -> float:
        model = self.get_model()
        model.eval()        
        metrics.reset(len(data_loader))

        references = []
        hypotheses = []

        print()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch = self.to_device(batch)
                # targets = batch["target"]
                logits, predicts, targets = model(batch)
                loss = self.criterion(logits, targets)

                metrics.update(predicts=predicts.squeeze(), targets=targets.squeeze(), loss=loss.item())

                if i % self.print_freq == 0:
                    print(f'Validation [{i + 1}/{len(data_loader)}]\t{metrics.format()}')

                references.extend(targets.squeeze())
                hypotheses.extend(logits.squeeze())

                if proof_of_concept:
                    break

            print(f'Validation [{i + 1}/{len(data_loader)}]\t{metrics.format()}')

            hypotheses = torch.from_numpy(np.stack(hypotheses, axis=0))
            references = torch.from_numpy(np.stack(references, axis=0))
            metrics.reset(len(data_loader))
            metrics.update(predicts=hypotheses, targets=references)
            print(f'\n* {metrics.format(show_average=False, show_batch_time=False, show_loss=False)}')

        return metrics.compute(hypotheses, references)
    
    def test(self, data_loader: DataLoader, metrics: Metrics, hook: Optional[str] = None, proof_of_concept: bool = False,
             return_logits: bool = False):
        model = self.get_model()
        model.eval()
        metrics.reset(len(data_loader))

        references = []
        hypotheses = []
        hypologits = []
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

                    logits, predicts, targets = model(batch)
                    handler.remove()

                    # print(activation[hook].shape)
                    activations.extend(activation[hook].squeeze().cpu().numpy())
                else:
                    logits, predicts, targets = model(batch)
               
                metrics.update(None, None)  # 

                if i % self.print_freq == 0:
                    print(f'Test [{i + 1}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')

                references.extend(targets.squeeze())
                hypotheses.extend(predicts.squeeze())
                if return_logits:
                    hypologits.extend(logits.squeeze())

                if proof_of_concept:
                    break

            print(f'Test [{i + 1}/{len(data_loader)}] {metrics.format(show_scores=False, show_loss=False)}')
            
            hypotheses = torch.Tensor(hypotheses)
            references = torch.Tensor(references)
            metrics.reset(len(data_loader))
            metrics.update(predicts=hypotheses, targets=references)
            print(f'\n* {metrics.format(show_average=False, show_batch_time=False, show_loss=False)}')

            metrics.compute(hypotheses, references)

        if hook is None:
            if return_logits:
                return hypotheses, references, torch.Tensor(hypologits), None
            else:
                return hypotheses, references, None, None
        if return_logits:
            return hypotheses, references, torch.Tensor(hypologits), np.array(activations)
        return hypotheses, references, None, np.array(activations)

    def fit(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader, metrics: Metrics,
            save_checkpoint: bool = True, proof_of_concept: bool = False):
        assert self.generator is not None

        epochs_since_improvement: int = self.epochs_since_improvement
        best_score: float = self.best_score

        for epoch in range(self.epoch, epochs):
            if epochs_since_improvement == self.epochs_early_stop:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % self.epochs_adjust_lr == 0:
                self.adjust_learning_rate(0.8)

            self.train(data_loader=train_loader, metrics=metrics, epoch=epoch, proof_of_concept=proof_of_concept)

            recent_score = self.valid(data_loader=valid_loader, metrics=metrics, proof_of_concept=proof_of_concept)
            
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
        