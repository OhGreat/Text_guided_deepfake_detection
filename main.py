import json
from argparse import Namespace
from os import makedirs
from os.path import join
from shutil import copy
from typing import Any, Tuple

import torch
import torchvision.transforms as T
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    OnExceptionCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from src.clip import clip
from src.clip.lightning_model import CLIPLightning
from src.utils.options import get_opts
from src.data.balanced_dataset import BalancedDS
from src.utils.utils import split_dataset, create_lighning_model


def create_trainer(
    visual_transforms: T.Compose, opts: Namespace
) -> Tuple[L.Trainer, DataLoader, DataLoader]:
    """Create Pytorch Lightning Trainer class."""

    if not opts.evaluate:
        # make sure only one of two options is defined
        assert (opts.valid_frac is not None) ^ (
            opts.real_eval is not None or opts.fake_eval is not None
        )
        if opts.valid_frac is not None:
            assert opts.valid_frac > 0
    else:
        # make sure we have paths to the evaluation datasets
        assert (opts.real_eval or opts.fake_eval) and not opts.valid_frac

    if opts.evaluate:
        valid_ds = BalancedDS(
            real_img_folders=opts.real_eval,
            fake_img_folders=opts.fake_eval,
            max_samples_per_class=opts.max_samples_per_class_eval,
            transform=visual_transforms,
            testing=True,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=opts.eval_batch_size,
            shuffle=False,
            num_workers=opts.workers,
        )
        train_dl = None

    else:
        train_ds = BalancedDS(
            real_img_folders=opts.real_train,
            fake_img_folders=opts.fake_train,
            max_samples_per_class=opts.max_samples_per_class_train,
            transform=visual_transforms,
        )

        if opts.valid_frac is not None:
            print("Splitting train set for validation.")
            train_ds, valid_ds = split_dataset(train_ds, opts.valid_frac, opts.seed)

        else:
            print("Validation data paths provided.")
            valid_ds = BalancedDS(
                real_img_folders=opts.real_eval,
                fake_img_folders=opts.fake_eval,
                max_samples_per_class=opts.max_samples_per_class_eval,
                transform=visual_transforms,
            )

        train_dl = DataLoader(
            train_ds,
            batch_size=opts.train_batch_size,
            shuffle=True,
            num_workers=opts.workers,
        )

        valid_dl = DataLoader(
            valid_ds,
            batch_size=opts.eval_batch_size,
            shuffle=False,
            num_workers=opts.workers,
        )

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=opts.ckpt_path,
        filename="weights-ep{epoch:02d}-val_loss{loss/val_clip_loss:.3f}",
        auto_insert_metric_name=False,
        every_n_train_steps=opts.ckpt_every_n_steps,
        save_on_train_epoch_end=True,
        save_top_k=-1,
        save_last=True,
    )

    exception_checkpoint_callback = OnExceptionCheckpoint(
        dirpath=opts.ckpt_path,
        filename="latest_interrupt",
    )

    early_stop_callback = EarlyStopping(
        monitor="loss/val_clip_loss",
        min_delta=0.00,
        patience=opts.early_stop,
        verbose=False,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)

    # setup logging directory
    if opts.log_dir is not None:
        opts.log_dir = join(opts.log_dir, opts.ckpt_path.split("/")[-1])
    else:
        opts.log_dir = join("./lightning_logs", opts.ckpt_path.split("/")[-1])
        print(f"Logs: {opts.log_dir}")
        opts.log_dir = (
            join("./evaluation_logs", opts.ckpt_path.split("/")[-1])
            if opts.evaluate
            else opts.log_dir
        )

    if opts.no_ckpt or opts.no_logger:
        callbacks = [
            early_stop_callback,
        ]
    else:
        callbacks = [
            checkpoint_callback,
            exception_checkpoint_callback,
            early_stop_callback,
            lr_monitor,
        ]

    # define trainer
    trainer = L.Trainer(
        logger=not opts.no_logger,
        default_root_dir=opts.log_dir,
        enable_checkpointing=True,
        accumulate_grad_batches=opts.accumulate_grad_batches,
        val_check_interval=opts.val_check_interval,
        callbacks=callbacks,
        max_epochs=opts.epochs,
        log_every_n_steps=1,
        limit_train_batches=opts.limit_train_batches,
        limit_val_batches=opts.limit_val_batches,
        reload_dataloaders_every_n_epochs=opts.reload_dataloaders_every_n_epochs,
        gradient_clip_val=opts.gradient_clip_val,
        accelerator="cpu" if opts.cpu else "gpu",
        precision=opts.mixed_precision,
    )
    return trainer, train_dl, valid_dl


def train(
    trainer: L.Trainer,
    l_model: L.LightningModule,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    opts: Namespace,
) -> None:
    """
    Main function for training, based on the Pytorch Linghtning trainer.
    """

    # create experiment directory and save all arguments + used descriptions
    makedirs(opts.ckpt_path, exist_ok=True)
    copy(opts.real_prompts, join(opts.ckpt_path, "real_descriptions.txt"))
    copy(opts.fake_prompts, join(opts.ckpt_path, "fake_descriptions.txt"))

    with open(join(opts.ckpt_path, "args.json"), "w") as res_f:
        json.dump(vars(opts), res_f, indent=2)

    if opts.resume is not None:
        if opts.load_only_model:
            print("Loading only model weights. (No optimizers will be loaded)")
            weights = torch.load(opts.resume)
            l_model.load_state_dict(weights["state_dict"])
            trainer.fit(l_model, train_dl, valid_dl)
        else:
            trainer.fit(l_model, train_dl, valid_dl, ckpt_path=opts.resume)
    else:
        trainer.fit(l_model, train_dl, valid_dl)

    return None


def evaluate(
    trainer: L.Trainer,
    l_model: L.LightningModule,
    valid_dl: DataLoader,
    opts: Namespace,
) -> dict:
    """
    Main function for evaluating, based on the Pytorch Linghtning trainer.
    """

    # Load model weights with interpolation possibility
    if opts.resume is not None:
        weights = torch.load(opts.resume)

        l_model.load_state_dict(weights["state_dict"])

        if opts.interpolate != "None":
            weights = l_model.model.state_dict()

            opts.interpolate = float(opts.interpolate)

            print("Interpolating finetuned weights with zero shot weights.")

            zero_shot_w = l_model.state_dict()

            assert set(weights.keys()) == set(zero_shot_w.keys())

            weights = {
                key: (1 - opts.interpolate) * zero_shot_w[key]
                + opts.interpolate * weights[key]
                for key in zero_shot_w.keys()
            }

        l_model.model.load_state_dict(weights)

    stats = trainer.test(
        model=l_model,
        dataloaders=valid_dl,
    )

    return stats


def main(opts: Namespace) -> Any:
    """
    Main model to run training and evaluation of CLIP configurations with Pytorch Lightning.

    Return:
    - None if training, a dictionary with evaluation results when evaluating.
    """

    l_model, visual_transforms = create_lighning_model(opts)

    # create dataloaders
    trainer, train_dl, valid_dl = create_trainer(
        visual_transforms=visual_transforms, opts=opts
    )

    # evaluate
    if opts.evaluate:
        ret = evaluate(
            trainer=trainer,
            l_model=l_model,
            valid_dl=valid_dl,
            opts=opts,
        )

    # train
    else:
        ret = train(
            trainer=trainer,
            l_model=l_model,
            train_dl=train_dl,
            valid_dl=valid_dl,
            opts=opts,
        )

    return ret


if __name__ == "__main__":
    opts = get_opts()
    main(opts)
