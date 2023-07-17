#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python main.py \
--clip_weights '/path/to/original/CLIP/weights' \
--resume 'path/to/saved/model/weights/ckpt.pt' \
--model 'ViT-L/14' \
--real_eval \
'/path/to/first/real/folder/eval/samples' \
'/path/to/second/real/folder/eval/samples' \
--fake_eval \
'/path/to/first/fake/folder/eval/samples' \
'/path/to/second/fake/folder/eval/samples' \
'/path/to/third/fake/folder/eval/samples' \
'/path/to/third/fake/folder/eval/samples' \
--real_prompts 'path/to/real/descriptions.txt' \
--fake_prompts 'path/to/fake/descriptions.txt' \
--max_samples_per_class 20000 \
--batch_size 16 \
--workers 4 \
--interpolate 'None' \
--seed 0 \
--evaluate;