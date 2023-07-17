# Text guided deepfake detection with CLIP

Robust text guided face forgery detector with CLIP. This repository is related to the work done for my Master's thesis during my internship at <a href="https://www.duckduckgoose.ai/">DuckDuckGoose</a>. The method is based on the original CLIP repository found <a href="https://github.com/openai/CLIP">here</a>.

## Install
The repository requires a `Python 3.10.8` installation with the packages specified in the `requirements.txt` file.
You can follow the instructions below to create the environment with Anaconda:

```bash
conda create --name clip python=3.10.8

conda activate clip

pip install -r requirements.txt
```

## Training & evaluating

The file `main.py` has been provided to train and evaluate configurations with pytorch-lightning. All the available command line arguments can be found in `src/utils/options.py`.

Example bash scripts for training and evaluating can be found under the `example_scripts` directory.

To train and evaluate models the oroginal CLIP weights are required to load the initial model.

The text descriptions for the real/fake classes should be stored in separate `.txt` files, with each line representing a description of the class.

The only structure required for the datasets is to have the real images and the fake images in different folders. Multiple folders can be used for each class.

For training, the datasets can be passed to the `--real_train` `--fake_train` `--real_eval` and `--fake_eval` Python arguments. The `--real_train` and `--fake_train` arguments are required and `--valid_frac` can be used to create validation.

For using the `multi_evaluator.py` script, a file with the dataset paths is required. The structure of this file can be found in `example_scripts/data_paths_example.json`.


## Loading and using a pretrained model

```python
import torch
from PIL import Image
from src.clip import clip
from src.utils.utils import load_model_from_lightning_ckpt

# path to lightning model weights
lightning_ckpt = "path/to/fine-tuned/lightning.ckpt"
# path to folder of original CLIP weights
clip_orig_path = "path/to/original/CLIP/weights/folder" 
# pathsd to the descriptions used for fine-tuning
real_prompts_path = "path/to/training/descriptions/real.txt"
fake_prompts_path = "path/to/training/descriptions/fake.txt"
# path to the image
face_img = Image.open("path/to/image.png")
# device to run inference
device = "cuda:0"

model, vision_transforms = load_model_from_lightning_ckpt(
    pkg_path=lightning_ckpt,
    clip_orig_path=clip_orig_path,
    device=device
)

# open descriptions
with open(real_prompts_path, "r") as f:
    real_texts = f.read().splitlines()
with open(fake_prompts_path, "r") as f:
    fake_texts = f.read().splitlines()

# tokenize descriptions
real_tok = clip.tokenize(real_texts)
fake_tok = clip.tokenize(fake_texts)
descriptions = torch.vstack((real_tok, fake_tok)).to(device)

# transform image to tensor
face_img = vision_transforms(face_img)
# unsqueeze to give batch dimension
face_img = face_img.unsqueeze(dim=0).to(device)

# inference step
emb_img, emb_txt, img_logits, txt_logits = model(face_img, descriptions)
# img_logits are what we are interested for the classification task
img_logits = img_logits.softmax(dim=-1)
print(f"% real: {img_logits[0,0]}, fake: {img_logits[0,1]}")
```