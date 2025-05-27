import os
import csv
from argparse import ArgumentParser


def generate_dataset_csv(data_root="data/example_dataset/rgb"):
    """
    Recursively walk through the 'data/flsea' directory,
    locate paired .tiff or .tif files in 'imgs/' and 'depth/' subfolders,
    then write separate CSVs for train and test data.

    Train/Test split is based on predefined folders:
      - Test: 'data/flsea/canyons/u_canyon', 'data/flsea/red_sea/sub_pier'
      - Train: All other directories

    Output CSV format (one row per matched pair):
        <img_path>,<depth_path>,<features_path>
    """

    out_csv = os.path.join(data_root, "inference.csv")

    # Temporary storage for file matching
    pairs = {}

    # Recursively search all subdirectories of flsea_root
    valid_exts = ('.tif', '.tiff', '.png', '.jpg')
    image_files = [
        f for f in os.listdir(data_root)
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(data_root, f))
    ]

    for file_name in image_files:
        full_path = os.path.join(data_root, file_name)
        base_name = os.path.splitext(file_name)[0]
        pairs.setdefault(base_name, {})["img"] = full_path

    # Open train and test CSV files for writing
    with open(out_csv, "w", newline="") as out_f:
        out_writer = csv.writer(out_f)

        for base_name, data_dict in sorted(pairs.items()):
            img_path = data_dict["img"]

            depth_path = img_path.replace("/rgb/", "/rgb/pred_depth/")
            depth_path = os.path.splitext(depth_path)[0] + "_depth.tif"

            features_path = img_path.replace("/rgb/", "/rgb/matched_features/")
            features_path = os.path.splitext(features_path)[0] + "_features.csv"

            segms_path = img_path.replace("/rgb/", "/rgb/matched_segms/")
            segms_path = os.path.splitext(segms_path)[0] + "_segms.npy"

            # Determine if the path belongs to test or train set
            out_writer.writerow([img_path, depth_path, features_path, segms_path])

    print(f"Inference data saved to {out_csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/example_dataset/rgb",
    )
    args = parser.parse_args()
    generate_dataset_csv(data_root=args.data_root)
