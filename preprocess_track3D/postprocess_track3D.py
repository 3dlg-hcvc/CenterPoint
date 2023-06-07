import json
import numpy as np
import open3d as o3d
import os
from egoexo import PointCloud
from egoexo.utils import read_json

INFERENCE_DIR = "/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/results_inference"
PATH_DICT = "/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/track3D/val/clip_dir.json"

DEBUG = False


# The obb visualization function from https://github.com/3dlg-hcvc/egoexo/blob/main/scripts/visualize_sequence.py
def get_obb_pose(obb):
    front = np.asarray(obb['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(obb['up'])
    up = up / np.linalg.norm(up)
    right = np.cross(up, front)
    orientation = np.eye(4)
    orientation[:3, :3] = np.stack([right, up, front], axis=0)
    translation = np.eye(4)
    translation[:3, 3] = -np.asarray(obb['centroid'])
    return np.dot(orientation, translation)

def get_o3d_obb_mesh(obb):
    obb_pose = get_obb_pose(obb)
    return o3d.geometry.OrientedBoundingBox(obb['centroid'], 
                                            obb_pose[:3, :3].T, 
                                            obb['axesLengths'])

def _mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        _mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    OUTPUT_DIR = f"{INFERENCE_DIR}/output"
    _mkdir_recursive(OUTPUT_DIR)

    with open(f"{INFERENCE_DIR}/results.json", "r") as f:
        paths = json.load(f)
    
    bbxes = np.load(f"{INFERENCE_DIR}/bbx.npy")
    scores = np.load(f"{INFERENCE_DIR}/score.npy")

    with open(PATH_DICT, "r") as f:
        path_dict = json.load(f)

    # Collect all the frames for the clips
    results = {}
    for i in range(len(paths)):
        path = paths[i]
        bbx = bbxes[i]
        score = scores[i]
        clip_index = path.split("/")[-3]
        if clip_index not in results:
            results[clip_index] = []
        results[clip_index].append((path, bbx, score))

    # Porcess each prediction
    for clip_index, preds in results.items():
        frame_num = len(preds)
        preds = sorted(preds, key=lambda x: x[0])
        # Make the folder
        original_path = path_dict[clip_index].split("raw_track3D/val/")[-1]
        _mkdir_recursive(f"{OUTPUT_DIR}/{original_path}/obb")
        
        # Load the transformation
        extrinsic_path = f"/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/track3D/val/{clip_index}/meta_data.json"
        with open(extrinsic_path, "r") as f:
            extrinsic = np.array(json.load(f)["mean_extrinsic"]).reshape((4, 4), order="F")

        if DEBUG:
            # Load the processed GT OBB
            processed_gt_obbs_path = f"/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/track3D/val/{clip_index}/box_info.json"
            with open(processed_gt_obbs_path, "r") as f:
                processed_gt_obbs = json.load(f)

        # Load the first-frame obb
        gt_obb_path = f"{path_dict[clip_index]}/obb/{preds[0][0].split('/')[-1].split('.')[0]}.json"
        with open(gt_obb_path, "r") as f:
            gt_obb = json.load(f)

        transformations = []
        for i in range(len(preds)):
            bbx = preds[i][1]
            score = preds[i][2]

            # Deal with the bbx parameters
            center = bbx[:3]
            angle = bbx[8]

            if DEBUG and False:
                processed_gt_obb = processed_gt_obbs[preds[i][0].split("/")[-1].split(".")[0]]
                center = processed_gt_obb[:3]
                angle = processed_gt_obb[8]

            if DEBUG and True:
                # Get the gt obb
                processed_gt_obb = processed_gt_obbs[preds[i][0].split("/")[-1].split(".")[0]]
                processed_gt_center = processed_gt_obb[:3]
                processed_gt_angle = processed_gt_obb[8]

                points_colors = np.load(preds[i][0])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_colors[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(points_colors[:, 3:])

                processed_gt_obb_mesh = o3d.geometry.OrientedBoundingBox(
                    center=processed_gt_center,
                    R=np.array(
                        [
                            [np.cos(processed_gt_angle), -np.sin(processed_gt_angle), 0],
                            [np.sin(processed_gt_angle), np.cos(processed_gt_angle), 0],
                            [0, 0, 1],
                        ]
                    ),
                    extent=np.array(processed_gt_obb[3:6]),
                )

                obb = o3d.geometry.OrientedBoundingBox(
                    center=center,
                    R=np.array(
                        [
                            [np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1],
                        ]
                    ),
                    extent=np.array(processed_gt_obb[3:6]),
                )

                o3d.visualization.draw_geometries([pcd, processed_gt_obb_mesh, obb])


            transformation = np.eye(4)
            transformation[:3, 3] = center
            transformation[:2, :2] = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            transformations.append(transformation)

        # Calculate the relative transformation
        pred_obbs = []
        for i in range(len(preds)):
            relative_transformation = np.dot(transformations[i], np.linalg.inv(transformations[0]))
            relative_transformation = np.dot(np.dot(np.linalg.inv(extrinsic), relative_transformation), extrinsic)
            pred_obb = {"centroid": [], "axesLengths": [], "front": [], "up": []}
            R = relative_transformation[:3, :3]
            t = relative_transformation[:3, 3]
            pred_obb["centroid"] = (R.dot(np.array(gt_obb['centroid'])) + t).tolist()
            pred_obb['up'] = R.dot(np.array(gt_obb['up'])).tolist() 
            pred_obb['front'] = R.dot(np.array(gt_obb['front'])).tolist() 
            pred_obb['axesLengths'] = gt_obb['axesLengths']

            if DEBUG and False:
                frame_index = preds[i][0].split('/')[-1].split('.')[0]
                gt_frame_path = f"{path_dict[clip_index]}"
                camera_path = f"{gt_frame_path}/camera/{frame_index}.json"
                depth_path = f"{gt_frame_path}/depth/{frame_index}.png"
                rgb_path = f"{gt_frame_path}/rgb/{frame_index}.png"
                obb_path = f"{gt_frame_path}/obb/{frame_index}.json"

                pcd = PointCloud()
                # The pcd_data already in the world coordinate
                pcd_data = pcd.rgbd2pcd(rgb_path, depth_path, camera_path)

                pred_bbx = get_o3d_obb_mesh(pred_obb)
                gt_bbx = get_o3d_obb_mesh(read_json(obb_path))
                o3d.visualization.draw_geometries([pcd_data, gt_bbx, pred_bbx])

            # Store the prediction
            # Infer the final json path
            path = preds[i][0]
            frame_index = path.split("/")[-1].split(".")[0]
            output_path = f"{OUTPUT_DIR}/{original_path}/obb/{frame_index}.json"
            with open(output_path, "w") as f:
                json.dump(pred_obb, f)

        

            

