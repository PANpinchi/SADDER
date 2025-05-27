import sys
import os

mmdet_path = os.path.abspath("../WaterMask")
sys.path.insert(0, mmdet_path)

import torch
from mmdet.apis.inference import init_detector, inference_detector
import numpy as np
import cv2
import mmcv
from tqdm import tqdm

# output shapes
out_height = 240
out_width = 320

# output
rel_folder = "matched_segms"


def vis_infer(
        checkpoints="./outputs_swin_base_ours/epoch_11.pth",
        config="./configs/_ours_/mask_rcnn_swin-b-p4-w7_fpn_1x_coco.py",
        data_dir='./data/tmp_imgs/',
        num_segms=30
):
    """
    Function to run the DetInferencer for visual inference and save segmentation masks.
    """
    model = init_detector(config, checkpoints, device='cuda:0')

    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.tiff'))]

    max_segms = 0
    total_segms = 0
    count = 0

    for img_name in tqdm(image_files):
        img_path = os.path.join(data_dir, img_name)
        try:
            result = inference_detector(model, img_path)

            # Extract segmentation masks
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result) if bbox_result else []
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ] if bbox_result else []
            labels = np.concatenate(labels) if labels else []

            # draw segmentation masks
            segms = None
            if segm_result is not None and len(labels) > 0:  # non empty
                segms = mmcv.concat_list(segm_result)
                if isinstance(segms[0], torch.Tensor):
                    segms = torch.stack(segms, dim=0).detach().cpu().numpy().astype(np.uint8)
                else:
                    segms = np.stack(segms, axis=0).astype(np.uint8)

                # Update max segmentation mask count
                max_segms = max(max_segms, segms.shape[0])
                total_segms += segms.shape[0]
                count += 1

                # Resize segmentation masks to output shape
                segms = np.array(
                    [cv2.resize(seg.astype(np.uint8), (out_width, out_height), interpolation=cv2.INTER_NEAREST) for seg
                     in segms])

                if segms.shape[0] > num_segms:
                    segms = segms[:num_segms]
                elif segms.shape[0] < num_segms:
                    padding = np.zeros((num_segms - segms.shape[0], out_height, out_width), dtype=np.uint8)
                    segms = np.vstack((segms, padding))

                # Save segmentation masks only if available
                if segm_result is not None:
                    out_dir = os.path.join(data_dir, rel_folder)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, img_name.replace(os.path.splitext(img_name)[-1], '_segms.npy'))
                    np.save(out_file, segms)
                else:
                    print('segm_result is None')
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return max_segms, total_segms, count


if __name__ == "__main__":
    data_dir_list = [
        '../uw_depth/data/flsea/canyons/flatiron/imgs/',
        '../uw_depth/data/flsea/canyons/horse_canyon/imgs/',
        '../uw_depth/data/flsea/canyons/tiny_canyon/imgs/',
        '../uw_depth/data/flsea/canyons/u_canyon/imgs/',
        '../uw_depth/data/flsea/red_sea/big_dice_loop/imgs/',
        '../uw_depth/data/flsea/red_sea/coral_table_loop/imgs/',
        '../uw_depth/data/flsea/red_sea/cross_pyramid_loop/imgs/',
        '../uw_depth/data/flsea/red_sea/dice_path/imgs/',
        '../uw_depth/data/flsea/red_sea/landward_path/imgs/',
        '../uw_depth/data/flsea/red_sea/northeast_path/imgs/',
        '../uw_depth/data/flsea/red_sea/pier_path/imgs/',
        '../uw_depth/data/flsea/red_sea/sub_pier/imgs/',
    ]

    config = {
        "checkpoints": "../WaterMask/outputs_swin_base_ours_pr2_old/epoch_11.pth",
        "config": "../WaterMask/configs/_ours_/ablation/mask_rcnn_swin-b-p4-w7_fpn_1x_coco_pr2.py",
        "data_dir": "../uw_depth/data/example_dataset/rgb/",
    }

    # FLSea
    # Max Segms:  66
    # Avg Segms:  17.626751281425435
    max_segms = 0
    total_segms = 0
    count = 0

    # curr_max_segms = vis_infer(
    #     checkpoints=config["checkpoints"],
    #     config=config["config"],
    #     data_dir=config["data_dir"],
    # )
    # max_segms = max(max_segms, curr_max_segms)

    for data_dir_name in data_dir_list:
        curr_max_segms, curr_total_segms, curr_count = vis_infer(
            checkpoints=config["checkpoints"],
            config=config["config"],
            data_dir=data_dir_name,
        )
        max_segms = max(max_segms, curr_max_segms)
        total_segms += curr_total_segms
        count += curr_count

    avg_segms = total_segms / count if count > 0 else 0
    print('Max Segms: ', max_segms)
    print('Avg Segms: ', avg_segms)
    print('Done!')
