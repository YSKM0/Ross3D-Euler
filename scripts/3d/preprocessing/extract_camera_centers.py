import argparse
import json
import os
import pickle

import numpy as np


def load_scene_meta(annotation_dir: str):
    scene = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl"), "rb") as f:
            data = pickle.load(f)["data_list"]
            for item in data:
                if item["sample_idx"].startswith("scannet"):
                    scene[item["sample_idx"]] = item
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", type=str, default="data/embodiedscan")
    parser.add_argument("--video_folder", type=str, default="data")
    parser.add_argument("--output_file", type=str, default="data/metadata/scannet_camera_centers.json")
    args = parser.parse_args()

    scene = load_scene_meta(args.annotation_dir)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    all_data = []

    for video_id in sorted(scene.keys()):
        meta_info = scene[video_id]
        axis_align_matrix = np.array(meta_info["axis_align_matrix"], dtype=np.float32)

        frame_files = [os.path.join(args.video_folder, img["img_path"]) for img in meta_info["images"]]
        camera_centers = []

        for frame_file in frame_files:
            pose_file = frame_file.replace("jpg", "txt")
            try:
                pose = np.loadtxt(pose_file).astype(np.float32)
            except Exception:
                if pose_file.split("/")[-1] != "view_0.txt":
                    raise
                pose = np.zeros((4, 4), dtype=np.float32)

            pose = axis_align_matrix @ pose
            camera_centers.append(pose[:3, 3].tolist())

        all_data.append({
            "video_id": video_id,
            "camera_centers": camera_centers,
        })

    with open(args.output_file, "w") as f:
        json.dump(all_data, f)

    print(f"Saved camera centers for {len(all_data)} scenes to {args.output_file}")


if __name__ == "__main__":
    main()
