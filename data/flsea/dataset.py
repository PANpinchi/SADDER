from depth_estimation.utils.data import (
    InputTargetDataset,
    InputDataset,
    IntPILToTensor,
    FloatPILToTensor,
    MutualRandomFactor,
    ReplaceInvalid,
    MutualRandomHorizontalFlip,
    # MutualRandomVerticalFlip,
)
from torchvision import transforms

import csv


def get_flsea_dataset(train=False, shuffle=False, device="cpu"):
    # filenames
    if train:
        index_file = "data/flsea/train.csv"
    else:
        index_file = "data/flsea/test.csv"
    lines = csv.reader(open(index_file).read().splitlines())
    rgb_depth_priors_tuples = [i for i in lines]

    # transforms
    if train:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
                transforms.ColorJitter(brightness=0.1, hue=0.05),
            ]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = transforms.Compose([MutualRandomHorizontalFlip()])
        target_samples_transform = transforms.Compose(
            [
                MutualRandomFactor(factor_range=(0.8, 1.2)),
            ]
        )

    # if not train
    else:
        input_transform = transforms.Compose(
            [IntPILToTensor(type="uint8", device=device)]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = None
        target_samples_transform = None

    dataset = InputTargetDataset(
        rgb_depth_priors_tuples=rgb_depth_priors_tuples,
        input_transform=input_transform,
        target_transform=target_transform,
        all_transform=all_transform,
        target_samples_transform=target_samples_transform,
        max_priors=200,
        shuffle=shuffle,
    )

    return dataset


def get_flsea_dataset_inference(priors=True):

    # filenames
    index_file = "data/flsea/test.csv"
    lines = csv.reader(open(index_file).read().splitlines())
    rgb_target_priors_tuples = [i for i in lines]

    # image transform
    image_transform = IntPILToTensor(type="uint8")

    # priors (or not)
    if priors:
        tuples = [[t[0], t[2], t[3]] for t in rgb_target_priors_tuples]  # omit target depth maps
        dataset = InputDataset(tuples, image_transform, max_priors=200)
    else:
        tuples = [[t[0], t[3]] for t in rgb_target_priors_tuples]  # omit target depth maps and priors
        dataset = InputDataset(tuples, image_transform, max_priors=0)

    return dataset
