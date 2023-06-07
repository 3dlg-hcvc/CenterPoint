import egoexo
from egoexo import PointCloud
from egoexo.utils import read_json
import open3d as o3d
import numpy as np
import glob
import argparse
import os
import multiprocessing
import torch
import json
from utils import (
    getCamera,
    matrix_to_quaternion,
    quaternion_to_matrix,
    get_o3d_obb_mesh,
    getArrowMesh,
)

DEBUG_MODE = False


def existDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_parser():
    parser = argparse.ArgumentParser(description="preprocess the data for track3D")
    parser.add_argument(
        "--data_dir",
        type=str,
        # default="/Users/shawn/Desktop/Research/proj-objtrack/CenterPoint/data/raw_track3D",
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/raw_track3D",
        help="The raw data path to the data",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="/Users/shawn/Desktop/Research/proj-objtrack/CenterPoint/data/track3D",
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/track3D",
        help="The raw data path to the data",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Create the output folder
    existDir(args.output_dir)
    existDir(f"{args.output_dir}/{args.split}")
    # Get the clip path for all data
    clip_paths = glob.glob(f"{args.data_dir}/{args.split}/*/clips/*/*")

    clip_num = []
    for clip_path in clip_paths:
        frame_path = glob.glob(f"{clip_path}/camera/*")
        clip_num.append(len(frame_path))

    clip_num = np.array(clip_num)
    import pdb

    pdb.set_trace()

