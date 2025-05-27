import torch
from torchvision.transforms.functional import hflip, vflip
from PIL import Image

import numpy as np

import pandas as pd
import random
from os.path import exists

from .depth_prior import get_depth_prior_from_features

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def visualize_and_save(parametrization, save_path):
    if isinstance(parametrization, torch.Tensor):
        parametrization = parametrization.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(2):
        axes[i].imshow(parametrization[i], cmap="inferno")
        axes[i].set_title(f"Parametrization Channel {i}")
        axes[i].axis("off")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def domain_projection(image: torch.Tensor) -> torch.Tensor:
    """
    Computes Domain Projection for underwater depth estimation.

    Args:
        image (torch.Tensor): A tensor of shape (3, H, W) representing an RGB image.
                              Values should be in the range [0, 1] or [0, 255].

    Returns:
        torch.Tensor: A depth map tensor of shape (H, W), normalized to the range [0, 1].
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("Input image must have shape (3, H, W).")

    # Extract R, G, B channels
    R = image[0, :, :]
    G = image[1, :, :]
    B = image[2, :, :]

    # Compute M = max(G, B) using PyTorch's element-wise maximum function
    M = torch.maximum(G, B)

    # Domain Projection formula using precomputed coefficients
    mu = torch.tensor([0.496, -0.389, 0.464], device=image.device)  # Ensure same device (CPU/GPU)
    depth = mu[0] + mu[1] * R + mu[2] * M  # Compute depth projection

    # Min-max normalization to scale depth values to [0, 1]
    depth_min, depth_max = depth.min(), depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)  # Avoid division by zero

    return depth  # Output shape: (H, W)


class InputTargetDataset:
    """Parameters:
    - rgb_depth_priors_tuples: List of filepath tuples of form (rgb, depth, sparse priors)
    - input_transform: Transform to apply to the input RGB image, returns torch Tensor
    - target_transform: Transform to apply to the target depth image, returns torch Tensor
    - all_transform: Transform to apply to both input, target and mask image (and depth samples as well), returns list of torch Tensors
    - target_samples_transform: Transfrom to apply to both target and depth samples, returns list of torch Tensors
    - max_priors: max number of priors to subsample
    - shuffle: shuffle dataset"""

    def __init__(
        self,
        rgb_depth_priors_tuples,
        input_transform,
        target_transform,
        all_transform=None,
        target_samples_transform=None,
        max_priors=200,
        shuffle=False,
    ) -> None:

        # file paths
        self.path_tuples = rgb_depth_priors_tuples

        # transforms applied to input and target
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.all_transform = all_transform

        # random shuffle tuples
        if shuffle:
            random.shuffle(self.path_tuples)

        # depth_samples
        self.target_samples_transform = target_samples_transform
        self.max_priors = max_priors

        # checking dataset for missing files
        if not check_dataset(self.path_tuples):
            print("WARNING, dataset has missing files!")
            # exit(1)

        print(f"Dataset with {len(self)} tuples.")

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, idx):

        # get filenames
        input_fn = self.path_tuples[idx][0]
        target_fn = self.path_tuples[idx][1]
        depth_samples_fn = self.path_tuples[idx][2]
        segms_samples_fn = self.path_tuples[idx][3]

        # --- TRY to open input_fn ---
        try:
            input_img = Image.open(input_fn).resize((640, 480))
        except OSError as e:
            # or "except Exception as e:" to catch all errors
            print(f"Error reading input image {input_fn}: {e}")
            print("Returning random substitution instead...")
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion fallback

        # --- TRY to open target_fn ---
        try:
            target_img = Image.open(target_fn).resize((320, 240), resample=Image.NEAREST)
        except OSError as e:
            print(f"Error reading target (depth) image {target_fn}: {e}")
            print("Returning random substitution instead...")
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion fallback

        # apply input/target transforms
        input_img = self.input_transform(input_img)
        target_img, mask = self.target_transform(target_img)

        # check if depth map has at least one valid value
        if not mask.any():
            print(
                f"File {target_fn} has no valid depth values, trying other image as substitution ..."
            )
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion

        # read sparse depth priors
        depth_samples = read_features(depth_samples_fn, self.max_priors, device=target_img.device)

        # check if features has at least one entry
        if depth_samples is None:
            print("Depth priors is None, trying other image as substitution ...")
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion

        # get dense parametrization from sparse priors
        parametrization = get_depth_prior_from_features(
            features=depth_samples.unsqueeze(0),  # add batch dimension
            height=240,
            width=320,
        ).squeeze(0)

        # --- TRY to open segms_samples_fn ---
        try:
            segms = np.load(segms_samples_fn)  # Load the .npy file
            segms = torch.tensor(segms, dtype=torch.float32, device=target_img.device)  # Convert to tensor
        except Exception as e:
            print(f"Error reading segmentation {segms_samples_fn}: {e}")
            print("Returning random substitution instead...")
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion fallback

        # apply target + prior transform
        if self.target_samples_transform is not None:
            target_img, parametrization, segms = self.target_samples_transform(
                [target_img, parametrization, segms]
            )

        # list of all output tensors
        tensor_list = [input_img, target_img, mask, parametrization, segms]

        # apply mutual transforms
        if self.all_transform is not None:
            tensor_list = self.all_transform(tensor_list)

        return tensor_list


def check_dataset(path_tuples):
        """Checks dataset for missing files."""
        for tuple in path_tuples:
            for f in tuple:
                if not exists(f):
                    print(f"Missing file: {f}.")
                    return False

        print(f"Checked {len(path_tuples)} tuples for existence, all ok.")

        return True

def read_features(path, max_priors, device="cpu"):
        """Read sparse priors from file and store in torch tensor."""

        # load samples (might be less than n_samples)
        depth_samples_data = pd.read_csv(path).to_numpy()

        # give warning when no features
        if len(depth_samples_data) == 0:
            print(f"WARNING: Features list {path} is empty, returning None!")
            return None
        else:
            rand_idcs = np.random.permutation(len(depth_samples_data))[
                : max_priors
            ]
            depth_samples = depth_samples_data[rand_idcs]  # select subset

        # tensor from numpy
        depth_samples = torch.from_numpy(depth_samples).to(device)

        return depth_samples


class InputDataset:
    """Similar to InputTargetDataset above, but for inference only. If priors are not available, set `max_priors` to zero."""

    def __init__(self, rgb_priors_segms_tuples, image_transform, max_priors=200) -> None:

        self.path_tuples = rgb_priors_segms_tuples
        self.image_transform = image_transform
        self.max_priors = max_priors

        if self.max_priors > 0:
            print(f"Using priors (max {self.max_priors} per image).")
            check_dataset(self.path_tuples)
        else:
            image_fns = [[t[0]] for t in self.path_tuples]
            print("Not using priors, using nullprior as placeholder.")
            check_dataset(image_fns)

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, idx):

        # get img filename
        input_fn = self.path_tuples[idx][0]

        # read img
        input_img = Image.open(input_fn).resize((640, 480))

        # apply image transforms to get tensor
        input_img = self.image_transform(input_img)

        # get keypoints filename
        depth_samples_fn = self.path_tuples[idx][1]

        # read sparse depth priors
        depth_samples = read_features(depth_samples_fn, self.max_priors, device=input_img.device)

        # get dense parametrization from sparse priors
        parametrization = get_depth_prior_from_features(
            features=depth_samples.unsqueeze(0),  # add batch dimension
            height=240,
            width=320,
        ).squeeze(0)

        segms_samples_fn = self.path_tuples[idx][2]
        try:
            segms = np.load(segms_samples_fn)  # Load the .npy file
            segms = torch.tensor(segms, dtype=torch.float32, device=input_img.device)  # Convert to tensor
        except FileNotFoundError:
            segms = torch.zeros((30, 240, 320), dtype=torch.float32, device=input_img.device)

        return [input_img, parametrization, segms]


class PreprocessInputDataset:
    """Similar to InputTargetDataset above, but for inference only. If priors are not available, set `max_priors` to zero."""

    def __init__(self, rgb_depth_priors_tuples, input_transform, target_transform) -> None:

        self.path_tuples = rgb_depth_priors_tuples
        self.input_transform = input_transform
        self.target_transform = target_transform

        image_fns = [[t[0]] for t in self.path_tuples]
        check_dataset(image_fns)

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, idx):

        # get filenames
        input_fn = self.path_tuples[idx][0]
        target_fn = self.path_tuples[idx][1]

        # --- TRY to open input_fn and target_fn ---
        input_img = Image.open(input_fn).resize((640, 480))
        target_img = Image.open(target_fn).resize((320, 240), resample=Image.NEAREST)

        # apply input/target transforms
        input_img = self.input_transform(input_img)
        target_img, _ = self.target_transform(target_img)

        # list of all output tensors
        tensor_list = [input_img, target_img]

        return tensor_list


class MutualRandomHorizontalFlip:
    """Randomly flips an input RGB imape and corresponding depth target horizontally with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):

        do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:

            for i in range(len(tensors)):

                tensors[i] = hflip(tensors[i])

        return tensors


class MutualRandomVerticalFlip:
    """Randomly flips an input RGB imape and corresponding depth target vertically with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):
        do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:
            for i in range(len(tensors)):

                tensors[i] = vflip(tensors[i])

        return tensors


class IntPILToTensor:
    """Converts an int PIL img to a torch float tensor in range [0,1]."""

    def __init__(self, type="uint8", custom_divider=None, device="cpu") -> None:

        self.device = device

        if type == "uint8":
            self.divider = 255
        elif type == "uint16":
            self.divider = 65535
        else:
            self.divider = 1

        if custom_divider is not None:
            self.divider = custom_divider  # ycb-video uses 10'000 as factor

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: ch x H x W
        if img_np.ndim == 3:
            img_np = img_np.transpose((2, 0, 1))
        elif img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        # convert to float and divide by set divider
        img_tensor = img_tensor.float().div(self.divider)

        return img_tensor


class FloatPILToTensor:
    """Converts a float PIL img to a torch float tensor"""

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: channels x height x width
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        return img_tensor


class MutualRandomFactor:
    """Multiply tensors by a random factor in given range."""

    def __init__(self, factor_range=(0.75, 1.25)) -> None:
        self.factor_range = factor_range

    def __call__(self, tensors):

        factor = (
            torch.rand(1).item() * (self.factor_range[1] - self.factor_range[0])
            + self.factor_range[0]
        )

        for i in range(len(tensors)):

            tensors[i][0, ...] *= factor

        return tensors


class ReplaceInvalid:
    """Replace invalid values (=0) of a tensor with a given vale."""

    def __init__(self, value=None):
        self.value = value

    def __call__(self, tensor):

        mask = get_mask(tensor)

        # if mask is empty, return None
        if not mask.any():
            # print(
            #     "Mask is empty, meaning all depth values invalid. Returning unchanged."
            # )

            return tensor, mask

        # change value of non valid pixels
        if self.value is not None:
            if self.value == "max":
                max = tensor[mask].max()
                tensor[~mask] = max
            elif self.value == "min":
                min = tensor[mask].min()
                tensor[~mask] = min
            else:
                tensor[~mask] = self.value

        return tensor, mask


def get_mask(depth):
    """Get mask depth > 0.0"""

    mask = depth.gt(0.0)

    return mask


def test_dataset():

    print("Testing InputTargetDataset class ...")

    # test specific imports
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from data.example_dataset.dataset import get_example_dataset

    dataset = get_example_dataset()

    # dataloader
    dataloader = DataLoader(dataset, batch_size=2)

    for batch_id, data in enumerate(dataloader):

        rgb_imgs = data[0]
        d_imgs = data[1]
        masks = data[2]
        parametrizations = data[3]

        for i in range(rgb_imgs.size(0)):

            rgb_img = rgb_imgs[i, ...]
            d_img = d_imgs[i, ...]
            mask = masks[i, ...]
            nn_parametrization = parametrizations[i, 0, ...].unsqueeze(0)
            prob_parametrization = parametrizations[i, 1, ...].unsqueeze(0)

            print(f"d range: [{d_img.min()}, {d_img.max()}]")

            plt.figure(f"rgb img {i}")
            plt.imshow(rgb_img.permute(1, 2, 0))
            plt.figure(f"d img {i}")
            plt.imshow(d_img.permute(1, 2, 0))
            plt.figure(f"mask {i}")
            plt.imshow(mask.permute(1, 2, 0))
            plt.figure(f"parametrization, NN {i}")
            plt.imshow(nn_parametrization.permute(1, 2, 0))
            plt.figure(f"parametrization, Probability {i}")
            plt.imshow(prob_parametrization.permute(1, 2, 0))

        plt.show()

        break  # only check first batch

    print("Testing DataSet class done.")


# run as "python -m depth_estimation.utils.data" from repo root
if __name__ == "__main__":
    test_dataset()
