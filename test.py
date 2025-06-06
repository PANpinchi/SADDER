import torch
from torch.utils.data import DataLoader

import numpy as np

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.loss import RMSELoss, SILogLoss, MARELoss

# from data.example_dataset.dataset import get_example_dataset
from data.flsea.dataset import get_flsea_dataset

############################################################
###################### CONFIG ##############################
############################################################

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = (
    # "../uw_depth/data/saved_models/model_e22_udfnet_lr0.0001_bs6_lrd0.9.pth"
    "saved_models/model_e24_udfnet_lr1e-05_bs6_lrd0.9.pth"
)

# DATASET = get_example_dataset(train=False, shuffle=False, device=DEVICE)
DATASET = get_flsea_dataset(
    train=False,
    shuffle=False,
    device=DEVICE,
)

############################################################
############################################################
############################################################


# losses
rmse_lin = RMSELoss()
rmse_log = SILogLoss(correction=0.0, scaling=1.0)
rmse_silog = SILogLoss(correction=1.0, scaling=1.0)
mare = MARELoss()

# max depth
dmax = []


@torch.no_grad()
def test():
    # device info
    print(f"Using device {DEVICE}")

    # model
    print(f"Loading model from {MODEL_PATH}")
    model = UDFNet(n_bins=80).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loading model done.")

    # dataloader
    dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE, drop_last=True)

    n_batches = len(dataloader)
    ranges = [None, 5.0, 1.0]
    rmse_lin_losses = [[], [], []]
    rmse_log_losses = [[], [], []]
    rmse_silog_losses = [[], [], []]
    mare_losses = [[], [], []]
    for batch_id, data in enumerate(dataloader):

        # inputs
        rgb = data[0].to(DEVICE)  # RGB image
        target = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        prior = data[3].to(DEVICE)  # precomputed features and depth values
        segms = data[4].to(DEVICE)  # precomputed features and depth values

        dmax.append(target.max().item())

        # nullprior
        # prior[:, :, :, :] = 0.0

        # null segms prior
        segms[:, :, :, :] = 0.0

        # outputs
        prediction, _ = model(rgb, prior, segms)

        # # enforce best fit at keypoint locations
        # priors_mask = (prior[:,1,:,:] == 1).unsqueeze(1)  # mask for location of keypoints, probability is ==1 at keypoint locations
        # scales = torch.zeros(BATCH_SIZE, 1, 1, 1).to(DEVICE)  # retrieve correction scale for best scale in RMSE sense
        # for i in range(BATCH_SIZE):
        #     scales[i] = get_scale(prediction[i], target[i], priors_mask[i])
        # prediction = scales * prediction

        # loss
        for i, r in enumerate(ranges):

            # for whole range select all
            if r is None:
                m_target = target[mask]
                m_prediction = prediction[mask]

            # for finite range select pixels with mask
            else:

                # mask for this range
                range_mask = target[mask] < r

                # skip if mask is empty
                if not range_mask.any():
                    continue

                m_target = target[mask][range_mask]
                m_prediction = prediction[mask][range_mask]

            # loss
            rmse_lin_losses[i].append(rmse_lin(m_prediction, m_target).item())
            rmse_log_losses[i].append(rmse_log(m_prediction, m_target).item())
            rmse_silog_losses[i].append(rmse_silog(m_prediction, m_target).item())
            mare_losses[i].append(mare(m_prediction, m_target).item())

        if batch_id % 10 == 0:
            print(f"{batch_id}/{n_batches}")

    for i, r in enumerate(ranges):
        print(f"Range: {r} m, using MEAN reduction:")
        print(f"RMSE (lin): {np.nanmean(rmse_lin_losses[i])}")
        print(f"RMSE (log): {np.nanmean(rmse_log_losses[i])}")
        print(f"RMSE (silog): {np.nanmean(rmse_silog_losses[i])}")
        print(f"MARE: {np.nanmean(mare_losses[i])}")
        print("---")
        print(f"Range: {r} m, using MEDIAN reduction:")
        print(f"RMSE (lin): {np.nanmedian(rmse_lin_losses[i])}")
        print(f"RMSE (log): {np.nanmedian(rmse_log_losses[i])}")
        print(f"RMSE (silog): {np.nanmedian(rmse_silog_losses[i])}")
        print(f"MARE: {np.nanmedian(mare_losses[i])}")
        print("\n===\n")

    print(f"max_depth mean: {np.mean(dmax)}")
    print(f"max_depth median: {np.median(dmax)}")


def get_scale(prediction, target, mask):
    # A*s = B
    A = prediction[mask].unsqueeze(1)
    B = target[mask].unsqueeze(1)
    s = torch.linalg.lstsq(A, B).solution

    return s


if __name__ == "__main__":
    test()
