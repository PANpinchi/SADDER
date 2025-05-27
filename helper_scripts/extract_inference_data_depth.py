"""
# > Script for inferencing UDepth on image/folder/video data
#    - Paper: https://arxiv.org/pdf/2209.12358.pdf
"""
import os
import sys
sys.path.append(os.path.abspath('.'))

import cv2
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
# local libs
from udepth_model.udepth import *
from utils.data import *
from utils.utils import *
from CPD.sod_mask import get_sod_mask
import tifffile
from tqdm import tqdm


def get_depth(image):
    """Generate depth map"""
    # Prepare SOD mask
    mask = np.array(get_sod_mask(image))
    # Convert RGB color space into RMI input space if needed
    if args.use_RMI:
        image = RGB_to_RMI(image)
    # Prepare data
    image_tensor = totensor(image).unsqueeze(0)
    input_img = torch.autograd.Variable(image_tensor.to(device=device))
    # Generate depth map
    _, out = net(input_img)
    # Apply guidedfilter to depth map
    result = output_result(out, mask)
    ab_result = output_ab_result(out, mask)

    return result, ab_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_img", type=str, default="./data/imgs/00002.png")
    parser.add_argument("--loc_folder", type=str, default="./data/example_dataset/rgb")
    parser.add_argument("--loc_video", type=str, default="./data/test_video.mp4")
    parser.add_argument("--model_RMI_path", type=str, default="./saved_udepth_model/model_RMI.pth")
    parser.add_argument("--model_RGB_path", type=str, default="./saved_udepth_model/model_RGB.pth")
    parser.add_argument("--use_RMI", action='store_true')
    parser.add_argument("--data_type", type=str, default="folder", choices=["image", "folder", "video"])
    args = parser.parse_args()

    # Define input space
    image_space = "RMI"  # if args.use_RMI else "RGB"

    # Create output folder if not exist
    output_folder = './data/example_dataset/rgb/pred_depth'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load specific model
    model_path = args.model_RMI_path if args.use_RMI else args.model_RGB_path
    net = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
    net.load_state_dict(torch.load(model_path))
    print("Model loaded: UDepth")

    net = net.to(device=device)
    net.eval()

    # Load data for image data type
    if args.data_type == "image":
        img_fn = args.loc_img.split("/")[-1]
        # Load data
        image = Image.open(args.loc_img)
        # Generate depth map
        result, ab_result = get_depth(image)
        # Save result
        plt.imsave(os.path.join(output_folder, img_fn), result, cmap='inferno')

    # Load data for folder data type
    if args.data_type == "folder":
        # Inferencing loop
        valid_exts = ('.tif', '.tiff', '.png', '.jpg')
        image_files = [
            f for f in os.listdir(args.loc_folder)
            if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(args.loc_folder, f))
        ]
        for img_fn in tqdm(image_files):
            # Load data
            img_path = os.path.join(args.loc_folder, img_fn)
            image = Image.open(img_path)
            # Generate depth map
            result, ab_result = get_depth(image)
            # Save result
            # plt.imsave(os.path.join(output_folder, img_fn), result, cmap='inferno')
            ab_result[ab_result > 15.] = 0.
            result_to_save = ab_result.astype(np.float32)
            output_path = os.path.join(output_folder, os.path.splitext(img_fn)[0] + "_depth.tif")
            tifffile.imwrite(output_path, result_to_save)
        print("Total images: {0}\n".format(len(image_files)))

    # Load data for video data type
    if args.data_type == "video":
        # Create output folder for results of video frames
        video_folder = os.path.join(output_folder, args.loc_video.split("/")[-1].split(".")[0])
        os.makedirs(video_folder)
        # Load the video
        vidObj = cv2.VideoCapture(args.loc_video)
        out_video_name = args.loc_video.split("/")[-1].replace(".", "_out.")
        success = True
        num = 1
        # Convert video into image frames and inference
        while (success):
            # Get image frame
            success, image = vidObj.read()
            if success:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Generate depth map
                result, ab_result = get_depth(Image.fromarray(image_rgb)) / 255
                # Save result
                plt.imsave(os.path.join(video_folder, "frame_{0}.jpg".format(num)), result, cmap='inferno')
                num += 1



