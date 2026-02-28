import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def uniform_sample_indices(total_frames: int, sample_count: int) -> List[int]:
    """
    Bit-for-bit mimic of repo uniform sampling:
      np.linspace(0, total_frames - 1, sample_count, dtype=int)

    Key behaviors:
    - Does NOT error when sample_count > total_frames (duplicates will occur).
    - Endpoints inclusive.
    - Cast-to-int truncation toward zero (same as numpy astype(int) for non-negative values).
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")

    if sample_count == 1:
        return [0]

    # Equivalent to np.linspace(0, total_frames - 1, sample_count, dtype=int)
    # linspace values: v_i = 0 + i * (total_frames-1) / (sample_count-1)
    # dtype=int truncates toward 0; for non-negative values this is floor.
    return [int(i * (total_frames - 1) / (sample_count - 1)) for i in range(sample_count)]


def pose_path_from_frame_file(frame_file: str) -> Path:
    """
    Match repo behavior described: pose file is in the same folder tree as the image,
    with only suffix replaced (.jpg/.png/etc -> .txt).
    """
    return Path(frame_file).with_suffix(".txt")


def load_pose(frame_file: str, allow_missing: bool = False):
    pose_path = pose_path_from_frame_file(frame_file)
    if not pose_path.exists():
        if allow_missing:
            return None, None
        raise FileNotFoundError(f"Missing pose file: {pose_path}")

    rows = []
    with open(pose_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])

    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        shape = (len(rows), len(rows[0]) if rows else 0)
        raise ValueError(f"Pose matrix must be 4x4, got {shape} at {pose_path}")

    camera_center = [rows[0][3], rows[1][3], rows[2][3]]
    return rows, camera_center


def build_entries(
    indices: List[int], frame_files: List[str], allow_missing_pose: bool
) -> List[Dict[str, Any]]:
    entries = []
    for idx in indices:
        # idx is always in [0, total_frames-1] due to construction of linspace endpoints.
        frame_file = frame_files[idx]
        c2w, center = load_pose(frame_file, allow_missing=allow_missing_pose)
        entries.append(
            {
                "index": idx,
                "frame_file": frame_file,
                "camera_to_world": c2w,
                "camera_center_world": center,
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Compare FVS-selected frames vs uniform sampling and export pose metadata."
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument(
        "--frame_sampling_strategy", type=str, default="uniform", choices=["uniform"]
    )
    parser.add_argument(
        "--allow_missing_pose",
        action="store_true",
        help="If set, write null pose fields when pose files are unavailable.",
    )
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        payload = json.load(f)

    if len(payload) != 1:
        raise ValueError("Expected input JSON to contain exactly one scene entry.")

    scene, scene_data = next(iter(payload.items()))
    total_frames = int(scene_data["total_frames"])
    fvs_indices = list(scene_data["selected_indices"])
    frame_files = list(scene_data["frame_files"])

    if len(frame_files) != total_frames:
        raise ValueError(
            f"frame_files length ({len(frame_files)}) must match total_frames ({total_frames})"
        )

    # Repo-accurate uniform sampling (duplicates allowed when sample_count > total_frames)
    uniform_indices = uniform_sample_indices(total_frames, args.sample_count)

    out = {
        "scene": scene,
        "total_frames": total_frames,
        "fvs": build_entries(
            fvs_indices, frame_files, allow_missing_pose=args.allow_missing_pose
        ),
        "uniform": build_entries(
            uniform_indices, frame_files, allow_missing_pose=args.allow_missing_pose
        ),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()