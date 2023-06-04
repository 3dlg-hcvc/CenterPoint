import json
import glob
import numpy as np

if __name__ == "__main__":
    dataset_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-objtrack/CenterPoint/data/track3D/train"
    meta_paths = glob.glob(f"{dataset_path}/*/meta_data.json")
    
    min_ranges = []
    max_ranges = []
    for meta_path in meta_paths:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        min_ranges.append(meta["min_range"])
        max_ranges.append(meta["max_range"])
    min_ranges = np.array(min_ranges)
    max_ranges = np.array(max_ranges)

    min_range = np.min(min_ranges, axis=0)
    max_range = np.max(max_ranges, axis=0)

    print(f"min: {min_range}")
    print(f"max: {max_range}")
    print(f"range: {max_range - min_range}")

