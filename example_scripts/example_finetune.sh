#!/bin/bash

CUDA_VISIBLE_DEVICES=2 \
python main.py \
--model "ViT-L/14" \
--clip_weights '/path/to/original/CLIP/weights' \
--real_train \
'/path/to/first/real/folder/train/samples' \
'/path/to/second/real/folder/train/samples' \
--fake_train \
'/path/to/first/fake/folder/train/samples' \
'/path/to/second/fake/folder/train/samples' \
'/path/to/third/fake/folder/train/samples' \
--real_eval \
'/path/to/first/real/folder/eval/samples' \
'/path/to/second/real/folder/eval/samples' \
--fake_eval \
'/path/to/first/fake/folder/eval/samples' \
'/path/to/second/fake/folder/eval/samples' \
'/path/to/third/fake/folder/eval/samples' \
--real_prompts 'path/to/real/descriptions.txt' \
--fake_prompts 'path/to/fake/descriptions.txt' \
--max_samples_per_class 5000 \
--batch_size 4 \
--workers 4 \
--reload_dataloaders_every_n_epochs 1 \
--accumulate_grad_batches 4 \
--ckpt_path 'path/to/save/model/checkpoints' \
--epochs 100 \
--lr 1e-6 \
--early_stop 10 \
--use_scheduler \
--first_cycle_steps 20000 \
--warmup_steps 10000 \
--gamma 0.5 \
--min_lr 0 \
--seed 1;