import os, glob
import json
import natsort

import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import box3d_overlap
# from box_util import gen

import argparse

THRESHOLD = 0.5

def load_obb(path):
    
    with open(path, "r") as fp:
        obb = json.load(fp)
    return obb

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

def obb_points_tensor(obb):

    obb_centroid = np.asarray(obb['centroid'])

    front = np.asarray(obb['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(obb['up'])
    up = up / np.linalg.norm(up)
    right = np.cross(up, front)

    lengths = np.asarray(obb['axesLengths'])
    d0 = right * lengths[0] / 2
    d1 = up * lengths[1] / 2
    d2 = front * lengths[2] / 2
    p0 = obb_centroid - d0 + d1 + d2
    p1 = obb_centroid - d0 + d1 - d2
    p2 = obb_centroid + d0 + d1 - d2
    p3 = obb_centroid + d0 + d1 + d2

    p4 = obb_centroid - d0 - d1 + d2
    p5 = obb_centroid - d0 - d1 - d2
    p6 = obb_centroid + d0 - d1 - d2
    p7 = obb_centroid + d0 - d1 + d2

    obb = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    return torch.from_numpy(obb.astype(np.float32))

# Calculate the angle between two rotation matrix (borrow from ANCSH paper)
def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def rot_diff_rad(rot1, rot2):
    theta = np.clip(( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2, a_min=-1.0, a_max=1.0)
    return np.arccos( theta ) % (2*np.pi)

def compute_pose_error(t_obb, p_obb):

    T_error = np.linalg.norm(np.array(t_obb['centroid']) - np.array(p_obb['centroid']))

    t_pose = get_obb_pose(t_obb)
    p_pose = get_obb_pose(p_obb)

    R_error = rot_diff_degree(t_pose[:3, :3], p_pose[:3, :3])

    return R_error, T_error

def main(args):
    
    sequences = [x for x in os.listdir(f'{args.gt_path}/')]

    clip_iou = []
    clip_ap = []
    clip_pose_error = []

    print(sequences)
    for seq in sequences:
        objects = os.listdir(os.path.join(args.gt_path, seq, 'clips'))
        for obj in objects:
            clips = os.listdir(os.path.join(args.gt_path, seq, 'clips', obj))
            for cl in clips:
                print(obj, cl)
                predicted_obbs = natsort.natsorted(glob.glob(os.path.join(args.pred_path, seq, 'clips', obj, cl, 'obb/*')))
                target_obbs = natsort.natsorted(glob.glob(os.path.join(args.gt_path, seq, 'clips', obj, cl, 'obb/*')))

                # print(predicted_obbs)
                # print(target_obbs)
                assert len(predicted_obbs) > 0
                assert len(predicted_obbs) == len(target_obbs), "number of predictions not equal"

                frame_wise_iou = []
                frame_wise_pose_err = []
                for t_path, p_path in zip(target_obbs, predicted_obbs):
                    
                    # load obb jsons
                    t_obb = load_obb(t_path)
                    p_obb = load_obb(p_path)

                    if len(t_obb) == 0 or len(p_obb) == 0:
                        frame_wise_iou.append(0)
                        # clip_ap
                        continue

                    # if p_obb['centroid'] is None:
                    #     import pdb; pdb.set_trace()
                    # print(p_obb)

                    # convert them to obb tensor with 8 points
                    tt_obb = obb_points_tensor(t_obb)
                    pt_obb = obb_points_tensor(p_obb)

                    # compute 3d-iou
                    _, iou_3d = box3d_overlap(tt_obb.unsqueeze(0), pt_obb.unsqueeze(0))
                    frame_wise_iou.append(iou_3d.item())

                    # compute pose error
                    R_error, T_error = compute_pose_error(t_obb, p_obb)
                    frame_wise_pose_err.append([R_error, T_error])

                frame_wise_iou = np.array(frame_wise_iou)

                # mean iou
                clip_iou.append(np.mean(frame_wise_iou))
                # ap 
                clip_ap.append(np.mean(frame_wise_iou > THRESHOLD))
                # pose error
                clip_pose_error.append(np.mean(frame_wise_pose_err, axis=0))

    clip_iou = np.array(clip_iou)
    clip_ap = np.array(clip_ap)
    clip_pose_error = np.array(clip_pose_error)

    print(f"MEAN - IOU: ", np.mean(clip_iou))
    print(f"MEAN - AP@{THRESHOLD}", np.mean(clip_ap))
    print(f"MEAN POSE ERROR: ", np.mean(clip_pose_error, axis=0))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True, 
                            help='path to directory with all the sequences with groundtruth annotations')
    parser.add_argument('--pred_path', type=str, required=True,
                            help='path to directory with all the sequences with predicted pose')
    args = parser.parse_args()

    main(args)