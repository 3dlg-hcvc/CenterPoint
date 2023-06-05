import sys
import pickle
import json
import random
import operator
import numpy as np
import glob
import os

from functools import reduce
from pathlib import Path
from copy import deepcopy

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main,
)
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class Track3DDataset(PointCloudDataset):
    NumPointFeatures = 6  # x, y, z, intensity, ring_index

    def __init__(
        self,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        load_interval=1,
        shuffle_points=False,
        **kwargs,
    ):
        self.load_interval = load_interval
        self.shuffle_points = shuffle_points
        # Get the point cloud path and the box info path
        self.pc_paths = glob.glob(f"{root_path}/*/pcd/*.npy")
        self.box_paths = glob.glob(f"{root_path}/*/box_info.json")
        super(Track3DDataset, self).__init__(
            root_path, None, pipeline, test_mode=test_mode, class_names=class_names
        )
        self._class_names = class_names
        self._num_point_features = Track3DDataset.NumPointFeatures
        # Load all box information and create the dictionary to index using the pc
        self.box_infos = {}
        for box_path in self.box_paths:
            with open(box_path, "r") as f:
                box_info = json.load(f)
            self.box_infos[os.path.dirname(box_path)] = box_info

    def reset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.pc_paths)

    def get_sensor_data(self, idx):
        info = self._nusc_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "virtual": self.virtual,
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        # pcd feature, x, y, z, r, g, b (rgb in 0-1)
        pc_path = self.pc_paths[idx]
        points = np.load(pc_path)
        if self.shuffle_points:
            np.random.shuffle(points)
        # Box information has been in cx, cy, cz, dx, dy, dz, vx, vy, yaw
        box_info = np.array(
            self.box_infos[os.path.dirname(os.path.dirname(pc_path))][
                os.path.basename(pc_path).split(".")[0]
            ]
        )
        res = {
            "points": points,
            "annotations": box_info,
        }
        data, _ = self.pipeline(res, None)
        return data

    def evaluation(self, detections, output_dir=None, testset=False):
        raise NotImplementedError
