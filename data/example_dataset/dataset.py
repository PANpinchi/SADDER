# this is a helper script that puts together datasets so they can be loaded
# with simpler getter functions for less repeated code in train/testing scripts

import csv

from depth_estimation.utils.data import (
    InputTargetDataset,
    InputDataset,
    PreprocessInputDataset,
    IntPILToTensor,
    FloatPILToTensor,
    MutualRandomFactor,
    ReplaceInvalid,
    MutualRandomHorizontalFlip,
    # MutualRandomVerticalFlip,
)
from torchvision import transforms


def get_example_dataset_preprocess(index_file="data/example_dataset/rgb/inference.csv", device="cpu"):
    # filenames
    lines = csv.reader(open(index_file).read().splitlines())
    rgb_target_priors_tuples = [i for i in lines]

    # image transform
    input_transform = transforms.Compose(
        [IntPILToTensor(type="uint8", device=device)]
    )
    target_transform = transforms.Compose(
        [
            FloatPILToTensor(device=device),
            ReplaceInvalid(value="max"),
        ]
    )

    # priors (or not)
    tuples = [[t[0], t[1]] for t in rgb_target_priors_tuples]  # omit target depth maps and priors
    dataset = PreprocessInputDataset(tuples,
                                     input_transform=input_transform,
                                     target_transform=target_transform)

    return dataset


def get_example_dataset_inference(index_file="data/example_dataset/rgb/inference.csv"):
    # filenames
    lines = csv.reader(open(index_file).read().splitlines())
    rgb_priors_segms_tuples = [i for i in lines]

    # image transform
    image_transform = IntPILToTensor(type="uint8")

    # priors (or not)
    tuples = [[t[0], t[2], t[3]] for t in rgb_priors_segms_tuples]  # omit target depth maps
    dataset = InputDataset(tuples, image_transform, max_priors=200)

    return dataset
