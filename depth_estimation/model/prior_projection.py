import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class PriorProjectionModule(nn.Module):
    """
    A module that processes MRI images through Max Pooling,
    Channel Attention, and outputs μ₀, μ₁, μ₂.
    """

    def __init__(self, num_feat=3, unsqueeze_factor=16):
        super(PriorProjectionModule, self).__init__()

        # Max Pooling Layer (reduces size by 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Channel Attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat * unsqueeze_factor, (1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat * unsqueeze_factor, num_feat, (1, 1), padding=0)
        )

        # Fully Connected Layer to predict [μ₀, μ₁, μ₂]
        self.fc = nn.Linear(num_feat, num_feat)  # Maps (N, 3) -> (N, 3)

        self.channel_attention = zero_module(self.channel_attention)
        self.fc = zero_module(self.fc)

    def forward(self, rgb):
        """
        Args:
            rgb: RGB input image, Nx3x480x640

        Returns:
            torch.Tensor: Predicted [μ₀, μ₁, μ₂] of shape (N, 3)
        """
        # Step 1: Max Pooling (Reduce resolution)
        rgb = self.maxpool(rgb)  # (N, 3, H//2, W//2)

        # Step 2: Channel Attention
        mri_weights = self.channel_attention(rgb).view(rgb.shape[0], -1)  # (N, 3)

        # Step 4: Fully Connected Layer (Predict μ₀, μ₁, μ₂)
        mu = self.fc(mri_weights).unsqueeze(2).unsqueeze(3)  # (N, 3, 1, 1)

        u0 = 0.49598983 + mu[:, 0]   # R
        u1 = -0.38875134 + mu[:, 1]  # M
        u2 = 0.46353632 + mu[:, 2]   # Constant

        # Step 5: Projection
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        gb_max = torch.maximum(g, b)
        torch_ones = torch.ones_like(r)
        depth_prior = u0 * r + u1 * gb_max + u2 * torch_ones
        depth_prior = torch.clamp(depth_prior, 0, 1)
        depth_prior = 1 - depth_prior.unsqueeze(1)

        return depth_prior
