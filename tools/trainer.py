import sys, os
sys.path.append(os.getcwd())

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ast import literal_eval
from copy import deepcopy
import logging
import os
import random
from typing import List, Dict

from tools.dataset import CustomImageDataset
from tools.get_region_feat import get_anatomical_region_data
from config import IMG_SIZE_FRC, batch_size, NUM_WORKERS
log = logging.getLogger(__name__)
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 1.0
SEED = 3407
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(dataset: str):
    mean = 0.471
    std = 0.302

    train_transforms = A.Compose(
        [

            A.LongestMaxSize(max_size=IMG_SIZE_FRC, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.PadIfNeeded(min_height=IMG_SIZE_FRC, min_width=IMG_SIZE_FRC, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['region_labels'])
    )

    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMG_SIZE_FRC, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMG_SIZE_FRC, min_width=IMG_SIZE_FRC, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['region_labels'])
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms
    

def get_datasets_as_dfs(dataset_path):
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_anatomicalfinding", "bbox_phrases"]

    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval, "bbox_anatomicalfinding": literal_eval, "bbox_phrases": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(dataset_path, dataset) + ".csv" for dataset in ["train", "valid"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")

    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]
    return datasets_as_dfs

def collate_fn(batch: List[Dict[str, Tensor]]):
    batch = list(filter(lambda x: x is not None, batch))

    image_shape = batch[0]["image"].size()
    images_batch = torch.empty(size=(len(batch), *image_shape))

    for i, sample in enumerate(batch):
        images_batch[i] = sample.pop("image")

    targets = batch

    batch_new = {}
    batch_new["images"] = images_batch
    batch_new["targets"] = targets

    return batch_new

def get_data_loaders(train_dataset, val_dataset):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def test_get_region_feat(val_loader):
    for batch_num, batch in enumerate(val_loader):
        images, targets = batch.values()
        batch_size = images.size(0)
        images = images.to(DEVICE, non_blocking=True)
        batch_list_dict = get_anatomical_region_data(images)
        print(batch_list_dict[0][0])
        input()


if __name__ == "__main__":
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    datasets_as_dfs = get_datasets_as_dfs("./data")
    train_dataset = CustomImageDataset(datasets_as_dfs["train"], transforms=get_transforms("train"))
    val_dataset = CustomImageDataset(datasets_as_dfs["valid"], transforms=get_transforms("valid"))

    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)
    test_get_region_feat(val_loader)