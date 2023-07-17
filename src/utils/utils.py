from argparse import Namespace
from typing import Tuple

import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as L

from src.clip.lightning_model import CLIPLightning
from src.clip.model import CLIP
import src.clip.clip as clip


def create_lighning_model(opts: Namespace) -> L.LightningModule:
    """
    Create the Pytorch Lightning module for fine-tuning CLIP.
    """

    # get model and preprocessing function for image and text
    model, visual_transforms = clip.load(
        opts.model,
        download_root=opts.clip_weights,
        load_weights=not opts.clip_from_scratch,
        reset_vision=opts.reset_vision,
    )

    # read prompt labels and preprocess
    with open(opts.real_prompts, "r") as f:
        real_texts = f.read().splitlines()

    with open(opts.fake_prompts, "r") as f:
        fake_texts = f.read().splitlines()
    
    l_model = CLIPLightning(
        model=model,
        real_prompts=real_texts,
        fake_prompts=fake_texts,
        only_vision=opts.only_vision,
        contrastive_margin=opts.contrastive_margin,
        topk=opts.topk,
        use_scheduler=opts.use_scheduler,
        sched_step_per_epoch=opts.sched_step_per_epoch,
        first_cycle_steps=opts.first_cycle_steps,
        warmup_steps=opts.warmup_steps,
        gamma=opts.gamma,
        min_lr=opts.min_lr,
        lr=opts.lr,
        weight_decay=opts.weight_decay,
    )

    return l_model, visual_transforms

def load_model_from_lightning_ckpt(
    pkg_path: str,
    clip_orig_path: str = None,
    model_name: str = "ViT-L/14",
    device: str = "cpu",
) -> CLIP:
    """
    Load a CLIP model from a fine-tuned Pytorch Lightning module.

    Args:
    - pkg_path: path to the Pytorch Linghtning checkpoint.
    - clip_orig_path: path to the original CLIP weights.
    - model_name: name of the pretrained CLIP architecture.
    - device: device where to run computations.
    """

    loaded = torch.load(pkg_path, map_location=device)

    # fix keys of dict to match keys in CLIP original model
    # else we get "model." prepended to each layer's name
    for key in list(loaded['state_dict'].keys()):
        loaded['state_dict'][key.split(".", 1)[-1]] = loaded['state_dict'].pop(key)

    model, vision_transforms = clip.load(
        name=model_name,
        download_root=clip_orig_path,
        device=device,
        load_weights=False,
        reset_vision=False,
    )

    model.load_state_dict(loaded['state_dict'])

    return model, vision_transforms

def split_dataset(
        dataset: Dataset,
        valid_frac: float = 0.05,
        random_split_seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
    """
    Splits a dataset in train and test.
    """
    
    train_size = int((1 - valid_frac) * len(dataset))
    valid_size = len(dataset) - train_size

    train_ds, valid_ds = random_split(
        dataset, 
        [train_size, valid_size], 
        generator = torch.Generator().manual_seed(random_split_seed)
    )
    print(f'Set 1: {len(train_ds)} samples, set 2: {len(valid_ds)} samples')

    return train_ds, valid_ds