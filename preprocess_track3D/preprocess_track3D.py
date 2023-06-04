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


def processClip(clip_path, output_path, clip_index):
    existDir(f"{output_path}/{clip_index}")

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
    obb_meshes = []
    obb_ups = []
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

        # Process the OBB data
        obb_data = read_json(obb_path)
        obb_ups.append(np.array(obb_data["up"]))
        obb_mesh = get_o3d_obb_mesh(obb_data)
        obb_meshes.append(obb_mesh)

        if DEBUG_MODE and False:
            # Visualize in this frame
            fx = pcd.intrinsic[0, 0]
            fy = pcd.intrinsic[1, 1]
            cx = pcd.intrinsic[0, 2]
            cy = pcd.intrinsic[1, 2]

            camera = getCamera(
                np.linalg.inv(pcd.extrinsic), fx, fy, cx, cy, scale=1, z_flip=True
            )
            up_arrow = getArrowMesh(
                np.array(obb_data["centroid"]),
                np.array(obb_data["centroid"]) + np.array(obb_data["up"]),
            )
            all_data.append(pcd_data)
            all_data += camera
            o3d.visualization.draw_geometries([pcd_data, obb_mesh, up_arrow] + camera)

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

    if DEBUG_MODE and False:
        # Visualize the mean camera pose
        camera = getCamera(
            np.linalg.inv(mean_extrinsic), fx, fy, cx, cy, scale=1, z_flip=True
        )
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
        all_data.append(coordinate)
        all_data += camera
        o3d.visualization.draw_geometries(all_data)

    # Transform all point cloud into the average camera coordinate
    assert len(pcds) == len(frame_names) == len(obb_meshes)
    if DEBUG_MODE:
        all_data = []

    min_range = None
    max_range = None

    prev_frame_center = None
    box_infos = {}

    for i in range(len(pcds)):
        # Transform the point cloud into the mean camera coordinate
        new_pcd_data = pcds[i].pcd.transform(mean_extrinsic)
        points = np.asarray(new_pcd_data.points)
        colors = np.asarray(new_pcd_data.colors)
        points_colors = np.concatenate([points, colors], axis=1)

        # Transform the obb into the camera coordinate system (the transform function is not implemented for OBB in open3d)
        new_obb_mesh = obb_meshes[i].rotate(mean_extrinsic[:3, :3], center=np.zeros(3))
        new_obb_mesh = new_obb_mesh.translate(mean_extrinsic[:3, 3])
        # Get the eight corner points to process to get the projected OBB (the format for the CenterPoint)
        obb_points = np.asarray(new_obb_mesh.get_box_points())
        # Then considering how to project it into 2D
        # Deal with the projection direction first
        z_min = np.min(obb_points[:, 2])
        z_max = np.max(obb_points[:, 2])
        z_center = (z_min + z_max) / 2
        dim_z = z_max - z_min
        # Then deal with the 2D plane stuff
        obb_points_2d = obb_points[:, 0:2]
        # Use the up direction as the projection direction
        obb_up = obb_ups[i]
        obb_up = obb_up / np.linalg.norm(obb_up)
        obb_up = np.dot(mean_extrinsic[:3, :3], obb_up)
        obb_up_2d = obb_up[0:2]
        if np.linalg.norm(obb_up_2d) == 0:
            # In case the up is parallel to the z-axis
            obb_up_2d = np.array([1, 0])
        else:
            obb_up_2d = obb_up_2d / np.linalg.norm(obb_up_2d)
        # Calculate the angle between +x and obb_up_2d
        angle = np.arccos(np.dot(obb_up_2d, np.array([1, 0])))
        assert angle >= 0 and angle <= np.pi
        # Judging the positive or negative of the angle based on the angle between up and +y
        if np.dot(obb_up_2d, np.array([0, 1])) < 0:
            angle = -angle

        # Rotate the obb points to calculate the range in x and y direciton, need to use the reverse versoin of the angle
        reverse_angle = -angle
        center_2d = (np.max(obb_points_2d, axis=0) + np.min(obb_points_2d, axis=0)) / 2
        obb_center = np.concatenate([center_2d, np.array([z_center])])
        # Do the rotation based on the center
        obb_points_2d = obb_points_2d - center_2d
        obb_points_2d = np.dot(
            np.array(
                [
                    [np.cos(reverse_angle), -np.sin(reverse_angle)],
                    [np.sin(reverse_angle), np.cos(reverse_angle)],
                ]
            ),
            obb_points_2d.T,
        ).T
        # Calculate the dim_x and dim_y
        dim_x = np.max(np.abs(obb_points_2d[:, 0])) * 2
        dim_y = np.max(np.abs(obb_points_2d[:, 1])) * 2

        # Caclulate the velocity in the projection plane
        if prev_frame_center is None:
            velocity = np.zeros(2)
        else:
            velocity = center_2d - prev_frame_center
        prev_frame_center = center_2d

        # Save the box info
        box_info = np.concatenate(
            [obb_center, np.array([dim_x, dim_y, dim_z]), velocity, np.array([angle])]
        )
        box_infos[frame_names[i]] = box_info.tolist()

        if min_range is None:
            min_range = np.min(points, axis=0)
            max_range = np.max(points, axis=0)
        else:
            min_range = np.min([min_range, np.min(points, axis=0)], axis=0)
            max_range = np.max([max_range, np.max(points, axis=0)], axis=0)

        # Save the points_colors data
        existDir(dir=f"{output_path}/{clip_index}/pcd")
        np.save(f"{output_path}/{clip_index}/pcd/{frame_names[i]}.npy", points_colors)

        if DEBUG_MODE and False:
            all_data.append(new_pcd_data)
            all_data.append(new_obb_mesh)
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
            up_arrow = getArrowMesh(
                np.array(new_obb_mesh.get_center()),
                np.array(new_obb_mesh.get_center() + obb_up),
            )

            # Reconstruct the projected version obb
            obb = o3d.geometry.OrientedBoundingBox(
                center=obb_center,
                R=np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1],
                    ]
                ),
                extent=np.array([dim_x, dim_y, dim_z]),
            )

            o3d.visualization.draw_geometries(
                [new_pcd_data, new_obb_mesh, obb, up_arrow, coordinate]
            )
    if DEBUG_MODE and False:
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries(all_data + [coordinate])

    # Store the annotation data into the json file
    with open(f"{output_path}/{clip_index}/box_info.json", "w") as f:
        json.dump(box_infos, f)

    # Store the meta_data to track
    meta_data = {
        "clip_path": clip_path,
        "mean_extrinsic": mean_extrinsic.reshape(16, order="F").tolist(),
        "min_range": min_range.tolist(),
        "max_range": max_range.tolist(),
    }

    with open(f"{output_path}/{clip_index}/meta_data.json", "w") as f:
        json.dump(meta_data, f)


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Create the output folder
    existDir(args.output_dir)
    existDir(f"{args.output_dir}/{args.split}")
    # Get the clip path for all data
    clip_paths = glob.glob(f"{args.data_dir}/{args.split}/*/clips/*/*")

    clip_dir = {}
    clip_index = 0
    for clip_path in clip_paths:
        processClip(clip_path, f"{args.output_dir}/{args.split}", clip_index)
        clip_dir[clip_index] = clip_path
        clip_index += 1

    # Save the clip dir
    with open(f"{args.output_dir}/{args.split}/clip_dir.json", "w") as f:
        json.dump(clip_dir, f)
