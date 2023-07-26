import torch
import pytorch_lightning as L
import torch.nn as nn
from random import randint
from typing import Any, List

from torch.nn.functional import cross_entropy
from torch.nn import CosineEmbeddingLoss
from pytorch_lightning.utilities import grad_norm
from torchmetrics import AUROC, Precision, Recall, Accuracy
from torchmetrics.classification import StatScores
from einops import rearrange

from src.utils.warmup_cosine_scheduler import CosineAnnealingWarmupRestarts
from src.clip import clip


class CLIPLightning(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        real_prompts: List[str],
        fake_prompts: List[str],
        use_scheduler: bool,
        sched_step_per_epoch: int,
        first_cycle_steps: int,
        warmup_steps: int,
        gamma: float,
        min_lr: float,
        lr: float,
        weight_decay: float,
        topk: int = None,
        class_weights: List[float] = [0.5, 0.5],
        only_vision: bool = False,
        contrastive_margin: float = None,
        *args: Any,
        **kwargs: Any,
    ) -> L.LightningModule:
        """
        Main Pytorch Lightning class used to fine-tune CLIP models.

        Args:
        - model: original CLIP model.
        - real_prompts: list of prompts for real class.
        - fake_prompts: list of prompts for fake class.
        - use_scheduler: activates warmup with cosine annealing scheduler.
        - lr: learning rate == peak learning rate when using scheduler.
        - min_lr: minimum learning rate to start the warmup.
        - sched_step_per_epoch: makes the step of the scheduler after every epoch instead of every batch (not recommended).
        - first_cycle_steps: total number of steps per learning rate cycle (cosine steps = first_cycle_steps - warmup_steps).
        - warmup_steps: number of warmup steps to reach lr.
        - gamma: decrease rate of max learning rate after the first cycle.
        - weight_decay: weight decay of the optimizer.
        - topk: Top-k sampling value to use when using multiple descriptions per class.
        - class_weights: class weights in the form of [real weight, fake weight]
        - only_vision: Freeze text encodeer and only train visual model.
        - contrastive_margin: margin for the contrastive loss. Higher values should make the contrastive learning less effective.

        Returns: Pytorch Lightning module to use for training.
        """

        super().__init__(*args, **kwargs)

        self.model = model

        # block gradients to text transformer if we only want to train the vision model
        self.model.transformer.requires_grad_(False if only_vision else True)

        self.register_buffer("real_prompts", clip.tokenize(real_prompts))
        self.register_buffer("fake_prompts", clip.tokenize(fake_prompts))

        # variables for itereating over prompts when using multiple per class when training
        self.real_len, self.fake_len = len(real_prompts) - 1, len(fake_prompts) - 1
        self.curr_real_idx, self.curr_fake_idx = 0, 0

        # these are used for the testing step
        self.register_buffer("prompt_feats", clip.tokenize(real_prompts + fake_prompts))
        self.prompt_split = len(real_prompts)

        # we always expect the real sample first and the fake one second
        self.register_buffer("labels", torch.tensor([0, 1]))
        self.topk = topk

        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float16))

        # contrastive loss
        self.contrastive_margin = contrastive_margin
        self.register_buffer("contrastive_target", torch.tensor([1]))
        self.contrastive_loss = CosineEmbeddingLoss(
            margin=contrastive_margin, reduction="none"
        )

        # optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.sched_step_per_epoch = "epoch" if sched_step_per_epoch else "step"
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.min_lr = min_lr
        self.only_vision = only_vision

        # metrics for evaluation and testing
        self.auc_fn = AUROC(task="multiclass", average="weighted", num_classes=2)
        self.prec_fn = Precision(task="multiclass", average="weighted", num_classes=2)
        self.acc_fn = Accuracy(task="multiclass", average="weighted", num_classes=2)
        self.rec_fn = Recall(task="multiclass", average="weighted", num_classes=2)
        self.scorer = StatScores(task="binary", num_classes=2)

    def configure_optimizers(self) -> Any:
        """Pytorch Lightning way of defining optimizer and schedulers."""
        params = (
            self.model.visual.parameters()
            if self.only_vision
            else self.model.parameters()
        )

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=self.weight_decay,
        )

        if not self.use_scheduler:
            return optimizer

        sched = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=self.first_cycle_steps,
            warmup_steps=self.warmup_steps,
            max_lr=self.lr,
            min_lr=self.min_lr,
            gamma=self.gamma,
        )

        scheduler = {
            "scheduler": sched,
            "interval": "step",
            "frequency": 1,
        }

        return ([optimizer], [scheduler])

    def on_before_optimizer_step(self, optimizer):
        """
        Pytorch Ligning function.
        Check for exploding graidents by analizing the L2 norm.
        Will log the gradients per layer in tensorboard.
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms, sync_dist=True)

    def training_step(self, batch, batch_idx) -> torch.tensor:
        """
        Pytorch Ligning function for training.
        """

        real, fake = batch

        labels = self.get_labels()

        real_feats_img, _, real_logits_per_image, real_logits_per_text = self.model(
            real, labels
        )
        fake_feats_img, _, fake_logits_per_image, fake_logits_per_text = self.model(
            fake, labels
        )

        img_logits = torch.stack((real_logits_per_image, fake_logits_per_image), dim=1)
        txt_logits = torch.stack((real_logits_per_text, fake_logits_per_text), dim=1)
        txt_logits = rearrange(txt_logits, "r f b -> b r f")
        # img_logits and txt_logits shape: (batch, 2, c, h, w)  -> 2: 1 real and 1 fake

        all_loss_img = [
            cross_entropy(img_couple, self.labels, self.class_weights) for img_couple in img_logits
        ]
        all_loss_txt = [
            cross_entropy(txt_couple, self.labels, self.class_weights) for txt_couple in txt_logits
        ]

        clip_loss = torch.stack(
            [
                (loss_img + loss_txt) / 2
                for loss_img, loss_txt in zip(all_loss_img, all_loss_txt)
            ]
        ).mean()

        if self.contrastive_margin is not None:
            contrastive_loss = self.contrastive_loss(
                real_feats_img, fake_feats_img, self.contrastive_target
            ).mean()

            self.log_dict(
                {
                    "loss/train_clip_loss": clip_loss,
                    "loss/train_contrsative_loss": contrastive_loss,
                    "loss/train_total_loss": clip_loss + contrastive_loss,
                },
                prog_bar=True,
                sync_dist=True,
            )

            return clip_loss + contrastive_loss

        self.log_dict(
            {"loss/train_clip_loss": clip_loss},
            prog_bar=True,
            sync_dist=True,
        )

        return clip_loss

    def validation_step(self, batch, batch_idx) -> torch.tensor:
        """Pytorch Ligning function for evaluating during training."""

        real, fake = batch

        # sample labels
        labels = self.get_labels()

        real_feats_img, _, real_logits_per_image, real_logits_per_text = self.model(
            real, labels
        )
        fake_feats_img, _, fake_logits_per_image, fake_logits_per_text = self.model(
            fake, labels
        )

        img_logits = torch.stack((real_logits_per_image, fake_logits_per_image), dim=1)
        txt_logits = torch.stack((real_logits_per_text, fake_logits_per_text), dim=1)
        txt_logits = rearrange(txt_logits, "r f b -> b r f")

        all_loss_img = [
            cross_entropy(img_couple, self.labels, self.class_weights) for img_couple in img_logits
        ]
        all_loss_txt = [
            cross_entropy(txt_couple, self.labels, self.class_weights) for txt_couple in txt_logits
        ]

        clip_loss = torch.stack(
            [
                (loss_img + loss_txt) / 2
                for loss_img, loss_txt in zip(all_loss_img, all_loss_txt)
            ]
        ).mean()

        if self.contrastive_margin is not None:
            contrastive_loss = self.contrastive_loss(
                real_feats_img, fake_feats_img, self.contrastive_target
            ).mean()

            self.log_dict(
                {
                    "loss/val_clip_loss": clip_loss,
                    "loss/val_contrsative_loss": contrastive_loss,
                    "loss/val_total_loss": clip_loss + contrastive_loss,
                },
                prog_bar=True,
                sync_dist=True,
            )

        self.log_dict(
            {
                "loss/val_clip_loss": clip_loss,
            },
            prog_bar=True,
            sync_dist=True,
        )

        for img_couple in img_logits:
            self.auc_fn.update(img_couple, self.labels)
            self.acc_fn.update(img_couple, self.labels)

        return clip_loss

    def on_validation_epoch_end(self) -> None:
        """Pytorch Ligning function."""
        stats = {
            "auc/val": self.auc_fn.compute(),
            "acc/val": self.acc_fn.compute(),
        }
        self.log_dict(stats, sync_dist=True)

        self.acc_fn.reset()
        self.auc_fn.reset()

    def test_step(self, batch, batch_idx) -> Any:
        """Pytorch Ligning function for testing."""
        x, y = batch

        _, _, logits_per_image, logits_per_text = self.model(x, self.prompt_feats)

        if self.topk is not None:
            topk_vals, topk_idx = torch.topk(logits_per_image, k=self.topk, dim=-1)
            real_idxes = topk_idx < self.prompt_split
            fake_idxes = topk_idx >= self.prompt_split

            real_logs = (topk_vals * real_idxes).sum(dim=-1) / real_idxes.sum(dim=-1)
            fake_logs = (topk_vals * fake_idxes).sum(dim=-1) / fake_idxes.sum(dim=-1)

        else:
            # when we are using multiple real/fake labels, we take the mean of the logits
            real_logs = (logits_per_image[:, : self.prompt_split]).mean(dim=-1)
            fake_logs = (logits_per_image[:, self.prompt_split :]).mean(dim=-1)

        logs = torch.stack((real_logs, fake_logs), dim=-1).softmax(dim=-1)

        self.auc_fn.update(logs, y)
        self.acc_fn.update(logs, y)
        self.prec_fn.update(logs, y)
        self.rec_fn.update(logs, y)
        self.scorer.update(logs.max(dim=-1)[-1], y)

    def on_test_epoch_end(self) -> dict:
        """Pytorch Ligning function."""
        scorer_stats = self.scorer.compute()

        stats = {
            "auc/test": self.auc_fn.compute(),
            "acc/test": self.acc_fn.compute(),
            "prec/test": self.prec_fn.compute(),
            "rec/test": self.rec_fn.compute(),
            "tp/real": scorer_stats[0],
            "fp/real": scorer_stats[1],
            "tp/fake": scorer_stats[2],
            "fp/fake": scorer_stats[3],
        }
        self.log_dict(stats, sync_dist=True,)

        self.acc_fn.reset()
        self.auc_fn.reset()
        self.prec_fn.reset()
        self.rec_fn.reset()

        return stats

    def get_labels(self) -> torch.tensor:
        """
        Returns appropriate labels for training or evaluation setting.
        """

        if self.real_len == 1 or self.real_len == 1:
            return self.prompt_feats

        if self.curr_real_idx >= self.real_len:
            self.curr_real_idx = 0
            idx = torch.randperm(self.real_prompts.size(0))
            self.real_prompts = self.real_prompts[idx]

        if self.curr_fake_idx >= self.fake_len:
            self.curr_fake_idx = 0
            idx = torch.randperm(self.fake_prompts.size(0))
            self.fake_prompts = self.fake_prompts[idx]

        labels = torch.stack(
            (
                self.real_prompts[self.curr_real_idx],
                self.fake_prompts[self.curr_fake_idx],
            )
        )

        self.curr_real_idx += 1
        self.curr_fake_idx += 1

        return labels
