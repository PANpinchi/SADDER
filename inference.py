from os.path import join
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Resize


from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_heatmap

from data.example_dataset.dataset import get_example_dataset_inference

from tqdm import tqdm


############################################################
###################### CONFIG ##############################
############################################################

BATCH_SIZE = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = (
    "saved_models/model_e24_udfnet_lr1e-05_bs6_lrd0.9_with_sadder.pth"
)
DATASET = get_example_dataset_inference(index_file="data/example_dataset/rgb/inference.csv")
OUT_PATH = "data/out_example"
SAVE = True

############################################################
############################################################
############################################################


@torch.no_grad()
def inference():

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

    total_time_per_image = 0.0
    n_batches = len(dataloader)
    for batch_id, data in enumerate(tqdm(dataloader)):

        # inputs
        rgb = data[0].to(DEVICE)  # RGB image
        prior = data[1].to(DEVICE)  # precomputed features and depth values
        segms = data[2].to(DEVICE)  # precomputed features and depth values

        # outputs
        start_time = time.time()
        prediction, _ = model(rgb, prior, segms)  # prediction in metric scale
        end_time = time.time()

        # time per img
        time_per_img = (end_time - start_time) / rgb.size(0)
        total_time_per_image += time_per_img

        # heatmap for visuals
        heatmap = gray_to_heatmap(prediction).to(DEVICE)  # for visualization

        # save outputs
        if SAVE:
            # resize = Resize(heatmap.size()[-2:])
            resize = Resize(rgb.size()[-2:])
            for i in range(rgb.size(0)):
                index = batch_id * BATCH_SIZE + i

                # out_rgb = join(OUT_PATH, f"{index}_rgb.png")
                # out_prediction = join(OUT_PATH, f"{index}_depth.png")
                out_heatmap = join(OUT_PATH, f"{index}".zfill(4) + "_heatmap.png")
                out_rgb_inputs = join(OUT_PATH, f"{index}".zfill(4) + "_inputs.png")

                # save_image(rgb[i], out_rgb)
                # save_image(prediction[i], out_prediction)
                save_image(heatmap[i], out_heatmap)
                save_image(rgb[i], out_rgb_inputs)

        # if batch_id % 10 == 0:
        #     print(f"{batch_id}/{n_batches}, {1.0/time_per_img} FPS")

    avg_time_per_image = total_time_per_image / n_batches
    avg_fps = 1.0 / avg_time_per_image

    print(f"Average time per image: {avg_time_per_image}")
    print(f"Average FPS: {avg_fps}")


if __name__ == "__main__":
    inference()