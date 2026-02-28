import argparse
import json
import logging
import math
import os
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    p1 = np.reshape(p1, (1, 3))
    return np.sum((p2 - p1) ** 2, axis=1)


def farthest_view_sampling(k: int, candidates, seed: int):
    points = np.asarray(candidates, dtype=float)
    n_points = points.shape[0]
    if n_points == 0 or k <= 0:
        return []
    if k > n_points:
        k = n_points

    dist = np.full(n_points, np.inf, dtype=float)
    point_left_idx = np.arange(n_points, dtype=int)
    selected_points = []

    np.random.seed(seed)
    selected_index = int(np.random.randint(0, n_points))
    selected_points.append(selected_index)
    point_left_idx = point_left_idx[point_left_idx != selected_index]

    for _ in range(1, k):
        if point_left_idx.size == 0:
            break

        active_point = points[selected_index]
        d = euclidean_dist(active_point, points[point_left_idx])
        dist[point_left_idx] = np.minimum(dist[point_left_idx], d)

        farthest_pos = int(np.argmax(dist[point_left_idx]))
        selected_index = int(point_left_idx[farthest_pos])
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, farthest_pos)

    return sorted(selected_points)


def expand_indices_to_target_count(indices, target_count: int):
    if target_count <= 0:
        return []
    if len(indices) == 0:
        return []
    if len(indices) == target_count:
        return indices
    if len(indices) > target_count:
        return indices[:target_count]

    pos = np.linspace(0, len(indices) - 1, target_count, dtype=int)
    return [indices[i] for i in pos]


def load_scene_meta(annotation_dir: str):
    scene = {}
    for split in ["train", "val", "test"]:
        p = os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl")
        with open(p, "rb") as f:
            data = pickle.load(f)["data_list"]
            for item in data:
                if item["sample_idx"].startswith("scannet"):
                    scene[item["sample_idx"]] = item
    return scene


def load_camera_centers(camera_center_file: str):
    if not camera_center_file or (not os.path.exists(camera_center_file)):
        logging.warning(
            "Camera center file not found (%s). Will compute camera centers on-the-fly from pose files.",
            camera_center_file,
        )
        return {}

    with open(camera_center_file, "r") as f:
        data = json.load(f)
    mapping = {}
    for dd in data:
        mapping[dd["video_id"]] = dd["camera_centers"]
    logging.info("Loaded camera centers for %d scenes from %s", len(mapping), camera_center_file)
    return mapping


def compute_camera_centers_for_scene(meta_info, video_folder: str):
    axis_align_matrix = np.array(meta_info["axis_align_matrix"], dtype=np.float32)
    frame_files = [os.path.join(video_folder, img["img_path"]) for img in meta_info["images"]]

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

    return frame_files, camera_centers


def load_yaml_video_ids(data_yaml: str, seed: int = 0):
    with open(data_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)

    datasets = yaml_data.get("datasets", [])
    all_samples = []

    for dataset in datasets:
        json_path = dataset.get("json_path")
        sampling_strategy = dataset.get("sampling_strategy", "all")
        sampling_number = None

        logging.info("Loading dataset %s with sampling strategy '%s'", json_path, sampling_strategy)

        if json_path.endswith(".jsonl"):
            cur_data = []
            with open(json_path, "r") as jf:
                for line in jf:
                    cur_data.append(json.loads(line.strip()))
        elif json_path.endswith(".json"):
            with open(json_path, "r") as jf:
                cur_data = json.load(jf)
        else:
            raise ValueError(f"Unsupported file type: {json_path}")

        if ":" in sampling_strategy:
            sampling_strategy, sampling_number = sampling_strategy.split(":")
            if "%" in sampling_number:
                sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data) / 100)
            else:
                sampling_number = int(sampling_number)

        if sampling_strategy == "first" and sampling_number is not None:
            cur_data = cur_data[:sampling_number]
        elif sampling_strategy == "end" and sampling_number is not None:
            cur_data = cur_data[-sampling_number:]
        elif sampling_strategy == "random" and sampling_number is not None:
            rng = random.Random(seed)
            rng.shuffle(cur_data)
            cur_data = cur_data[:sampling_number]

        logging.info("Loaded %d samples from %s", len(cur_data), json_path)
        all_samples.extend(cur_data)

    video_ids = [item["video"] for item in all_samples if "video" in item]
    unique_video_ids = sorted(set(video_ids))

    logging.info(
        "Collected %d total samples, %d samples with video field, %d unique video ids",
        len(all_samples), len(video_ids), len(unique_video_ids)
    )
    return unique_video_ids


def main():
    parser = argparse.ArgumentParser(description="Precompute and cache FVS selections for scenes listed by a training YAML")
    parser.add_argument("--data_yaml", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default="data/embodiedscan")
    parser.add_argument("--video_folder", type=str, default="data")
    parser.add_argument("--camera_center_file", type=str, default="data/metadata/scannet_camera_centers.json")
    parser.add_argument("--fvs_cache_file", type=str, default="data/metadata/scannet_fvs_selected_frames.json")
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument(
        "--save_camera_centers",
        action="store_true",
        help="Save computed camera centers back to --camera_center_file for future reuse.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(asctime)s] %(levelname)s %(message)s")

    logging.info("Starting FVS pre-sampling")
    scene = load_scene_meta(args.annotation_dir)
    logging.info("Loaded EmbodiedScan metadata for %d ScanNet scenes", len(scene))
    camera_centers = load_camera_centers(args.camera_center_file)
    target_video_ids = load_yaml_video_ids(args.data_yaml, seed=args.seed)

    num_frames_to_sample = args.frames_upbound if args.force_sample else 10

    cache = {}
    if os.path.exists(args.fvs_cache_file):
        try:
            with open(args.fvs_cache_file, "r") as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            logging.warning("Existing FVS cache is invalid JSON (%s); starting from empty cache", args.fvs_cache_file)
            cache = {}

    cache_hits = 0
    cache_misses = 0
    skipped = 0

    for video_id in tqdm(target_video_ids, desc="Pre-sampling FVS scenes"):
        if video_id not in scene:
            logging.warning("Skipping %s: not in EmbodiedScan scene metadata", video_id)
            skipped += 1
            continue

        meta_info = scene[video_id]
        frame_files = [os.path.join(args.video_folder, img["img_path"]) for img in meta_info["images"]]

        if video_id in camera_centers:
            cur_centers = camera_centers[video_id]
        else:
            frame_files_from_pose, cur_centers = compute_camera_centers_for_scene(meta_info, args.video_folder)
            if frame_files_from_pose != frame_files:
                logging.warning("Skipping %s due to frame file mismatch between metadata and pose scan", video_id)
                skipped += 1
                continue
            camera_centers[video_id] = cur_centers

        if len(cur_centers) != len(frame_files):
            logging.warning(
                "Skipping %s due to length mismatch (centers=%d, frames=%d)",
                video_id, len(cur_centers), len(frame_files),
            )
            skipped += 1
            continue

        entry = cache.get(video_id)
        is_cache_hit = (
            isinstance(entry, dict)
            and entry.get("total_frames") == len(frame_files)
            and entry.get("frame_files") == frame_files
            and isinstance(entry.get("selected_indices"), list)
            and len(entry["selected_indices"]) == num_frames_to_sample
            and all(isinstance(i, int) and 0 <= i < len(frame_files) for i in entry["selected_indices"])
        )

        if is_cache_hit:
            cache_hits += 1
            continue

        selected_indices = farthest_view_sampling(
            k=num_frames_to_sample,
            candidates=np.asarray(cur_centers, dtype=float),
            seed=args.seed,
        )
        selected_indices = expand_indices_to_target_count(selected_indices, num_frames_to_sample)
        cache[video_id] = {
            "total_frames": len(frame_files),
            "selected_indices": selected_indices,
            "selected_frame_files": [frame_files[i] for i in selected_indices],
            "frame_files": frame_files,
        }
        cache_misses += 1

    cache_dir = os.path.dirname(args.fvs_cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="fvs_cache_", suffix=".json", dir=cache_dir or None)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, args.fvs_cache_file)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if args.save_camera_centers and args.camera_center_file:
        camera_center_dir = os.path.dirname(args.camera_center_file)
        if camera_center_dir:
            os.makedirs(camera_center_dir, exist_ok=True)
        camera_center_list = [
            {"video_id": vid, "camera_centers": centers}
            for vid, centers in sorted(camera_centers.items())
        ]
        with open(args.camera_center_file, "w") as f:
            json.dump(camera_center_list, f)
        logging.info("Saved camera centers for %d scenes to %s", len(camera_center_list), args.camera_center_file)

    logging.info("Finished FVS pre-sampling")
    logging.info("Scenes requested: %d", len(target_video_ids))
    logging.info("Cache hits: %d", cache_hits)
    logging.info("Cache misses (recomputed): %d", cache_misses)
    logging.info("Skipped scenes: %d", skipped)
    logging.info("Final cache entries: %d", len(cache))
    logging.info("Saved cache file to: %s", args.fvs_cache_file)


if __name__ == "__main__":
    main()
