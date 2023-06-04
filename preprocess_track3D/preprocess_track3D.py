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
from utils import getCamera, matrix_to_quaternion, quaternion_to_matrix

DEBUG_MODE = True


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


def processClip(clip_path, output_path):
    camera_base_path = f"{clip_path}/camera"
    depth_base_path = f"{clip_path}/depth"
    rgb_base_path = f"{clip_path}/rgb"
    obb_base_path = f"{clip_path}/obb"

    # Extract all frame names
    camera_paths = glob.glob(f"{camera_base_path}/*.json")
    frame_names = sorted(
        [os.path.basename(path).split(".")[0] for path in camera_paths]
    )

    pcds = []
    if DEBUG_MODE:
        all_data = []

    for frame_name in frame_names:
        camera_path = f"{camera_base_path}/{frame_name}.json"
        depth_path = f"{depth_base_path}/{frame_name}.png"
        rgb_path = f"{rgb_base_path}/{frame_name}.png"
        obb_path = f"{obb_base_path}/{frame_name}.json"

        pcd = PointCloud()
        # The pcd_data already in the world coordinate
        pcd_data = pcd.rgbd2pcd(rgb_path, depth_path, camera_path)
        # Record all relevant information
        pcds.append(pcd)
        if DEBUG_MODE and True:
            # Visualize in this frame
            fx = pcd.intrinsic[0, 0]
            fy = pcd.intrinsic[1, 1]
            cx = pcd.intrinsic[0, 2]
            cy = pcd.intrinsic[1, 2]

            camera = getCamera(
                np.linalg.inv(pcd.extrinsic), fx, fy, cx, cy, scale=1, z_flip=True
            )
            all_data.append(pcd_data)
            # all_data += camera
            # o3d.visualization.draw_geometries([pcd_data] + camera)

    if DEBUG_MODE and False:
        o3d.visualization.draw_geometries(all_data)

    # Caclualte the mean camera extrinsic matrix (world -> camera) to used as the projection coordinate
    extrinsics = np.array([pcd.extrinsic for pcd in pcds])
    mean_translation = np.mean(extrinsics[:, 0:3, 3], axis=0)
    rotation_matrices = extrinsics[:, 0:3, 0:3]
    quaternions = np.array(matrix_to_quaternion(torch.tensor(rotation_matrices)))
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_rotation_matrix = quaternion_to_matrix(torch.tensor(mean_quaternion))
    mean_extrinsic = np.eye(4)
    mean_extrinsic[0:3, 0:3] = mean_rotation_matrix
    mean_extrinsic[0:3, 3] = mean_translation

    if DEBUG_MODE and True:
        # Visualize the mean camera pose
        camera = getCamera(
                np.linalg.inv(mean_extrinsic), fx, fy, cx, cy, scale=1, z_flip=True
            )
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
        all_data.append(coordinate)
        all_data += camera
        o3d.visualization.draw_geometries(all_data)

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
