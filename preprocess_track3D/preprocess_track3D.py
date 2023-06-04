import egoexo
from egoexo import PointCloud
from egoexo.utils import read_json
import open3d as o3d
import numpy as np
import glob
import argparse
import os
import multiprocessing

def existDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def get_parser():
    parser = argparse.ArgumentParser(description="preprocess the data for track3D")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/shawn/Desktop/Research/proj-objtrack/CenterPoint/data/raw_track3D",
        # default="/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/raw_track3D",
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
        default="/Users/shawn/Desktop/Research/proj-objtrack/CenterPoint/data/track3D",
        # default="/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/track3D",
        help="The raw data path to the data",
    )
    return parser

def processClip(clip_path, output_path):
    camera_base_path = f"{clip_path}/camera"
    depth_base_path = f"{clip_path}/depth"
    rgb_base_path = f"{clip_path}/rgb"
    obb_base_path = f"{clip_path}/obb"

    # Extract all frame names
    camera_paths = glob.glob(f"{camera_base_path}/*.json")
    frame_names = sorted([os.path.basename(path).split(".")[0] for path in camera_paths])

    for frame_name in frame_names:
        camera_path = f"{camera_base_path}/{frame_name}.json"
        depth_path = f"{depth_base_path}/{frame_name}.png"
        rgb_path = f"{rgb_base_path}/{frame_name}.png"
        obb_path = f"{obb_base_path}/{frame_name}.json"

        
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    args = get_parser().parse_args()
    # Create the output folder
    existDir(args.output_dir)
    # Get the clip path for all data
    clip_paths = glob.glob(f"{args.data_dir}/{args.split}/*/clips/*/*")

    for clip_path in clip_paths:
        processClip(clip_path, args.output_dir)

    import pdb
    pdb.set_trace()
