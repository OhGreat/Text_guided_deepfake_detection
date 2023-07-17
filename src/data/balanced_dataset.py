import re
from os import listdir
from os.path import join
from random import sample, shuffle
from typing import List

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class BalancedDS(Dataset):
    """Dataset class that can balance the number of samples in unbalanced data folders."""

    def __init__(
        self,
        real_img_folders: List[str],
        fake_img_folders: List[str],
        max_samples_per_class: int = None,
        max_tot_samples: int = None,
        transform: T.Compose = None,
        exts: List[str] = ["jpg", "jpeg", "png"],
        testing: bool = False,
    ) -> None:
        """
        Args:
        - real_img_folders: list of folder paths for real images.
        - fake_img_folders: list of folder paths for fake images.
        - max_samples_per_class: number of maximum samples per class to use.
        - max_tot_samples: sub-samples after having created the real and fake datasets (can end up with unvbalanced sets).
        - transform: visual transforms required by the model.
        - exts: extensions of files to consider.
        - testing: set to True for inference after training.
        """
        super().__init__()

        # PIL fix for big images
        Image.MAX_IMAGE_PIXELS = 933120000

        self.real_folders = real_img_folders
        self.fake_folders = fake_img_folders

        self.max_tot_samples = max_tot_samples

        self.transform = transform

        # collect samples as list of lists for each folder.

        self.real_paths = []
        self.fake_paths = []

        self.testing = testing

        # filter files by extension

        patterns = [r".*\." + re.escape(ext) + "$" for ext in exts]

        for i, folder in enumerate(real_img_folders):
            fold_paths = []
            for filename in listdir(folder):
                for pat in patterns:
                    if re.match(pat, filename):
                        fold_paths.append(join(folder, filename))

            self.real_paths.append(fold_paths)
        real_lens = [len(ds) for ds in self.real_paths]

        for i, folder in enumerate(fake_img_folders):
            fold_paths = []
            for filename in listdir(folder):
                for pat in patterns:
                    if re.match(pat, filename):
                        fold_paths.append(join(folder, filename))

            self.fake_paths.append(fold_paths)
        fake_lens = [len(ds) for ds in self.fake_paths]

        print("real folder lens:", real_lens, "fake folder lens:", fake_lens)

        # calculate maximum number of samples for each dataset, to balance classes.

        if len(fake_lens) == 0:
            min_len = min(real_lens)
            print("No fake samples found")
        elif len(real_lens) == 0:
            min_len = min(fake_lens)
            print("No real samples found, min fake:", min_len)
        elif len(fake_lens) == 0 and len(real_lens) == 0:
            exit("Badly defined dataset folders. Non samples found.")
        else:
            min_len = min(min(real_lens), min(fake_lens))

        if len(real_lens) > 0:
            max_per_folder = max_samples_per_class // len(real_lens)
            samp_folders = max_per_folder if max_per_folder < min_len else min_len
            print(
                f"Real per folder: {samp_folders}, total: {samp_folders*len(real_lens)}"
            )

            real_samps = samp_folders  # // len(real_lens)
            self.real_paths = [sample(ds, real_samps) for ds in self.real_paths]

            # collapse all real paths to a 1-dimensional array
            paths_ = []
            for curr_folder, dataset in enumerate(self.real_paths):
                for idx_img, sample_path in enumerate(dataset):
                    paths_.append((sample_path, 0))
                print(
                    f"Using {idx_img+1} real samples from {self.real_folders[curr_folder]}"
                )
            self.real_paths = paths_

        if len(fake_lens) > 0:
            max_per_folder = max_samples_per_class // len(fake_lens)
            samp_folders = max_per_folder if max_per_folder < min_len else min_len

            print(
                f"Fake per folder: {samp_folders}, total: {samp_folders*len(fake_lens)}"
            )

            fake_samps = samp_folders
            self.fake_paths = [sample(ds, fake_samps) for ds in self.fake_paths]

            # collapse all fake paths to a 1-dimensional array
            paths_ = []
            for curr_folder, dataset in enumerate(self.fake_paths):
                for idx_img, sample_path in enumerate(dataset):
                    paths_.append((sample_path, 1))
                print(
                    f"Using {idx_img+1} fake samples from {self.fake_folders[curr_folder]}"
                )
            self.fake_paths = paths_

        # self.all_paths =  (self.real_paths + self.fake_paths)
        if not self.testing:
            final_len = min(len(self.real_paths), len(self.fake_paths))
            self.real_paths = self.real_paths[:final_len]
            self.fake_paths = self.fake_paths[:final_len]
            shuffle(self.real_paths)
            shuffle(self.fake_paths)
        else:
            self.all_paths = self.real_paths + self.fake_paths
            shuffle(self.all_paths)

    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self.all_paths) if self.testing else len(self.real_paths)

    def __getitem__(self, idx: int) -> torch.tensor:
        """Return data in required format."""

        if self.testing:
            path, label = self.all_paths[idx]
            img = self.transform(Image.open(path))
            return img, label

        real_path, real_label = self.real_paths[idx]
        real_img = self.transform(Image.open(real_path))

        fake_path, fake_label = self.fake_paths[idx]
        fake_img = self.transform(Image.open(fake_path))

        return (real_img, fake_img)
