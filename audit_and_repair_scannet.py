#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAW_SUFFIXES = [
    "{scene}.sens",
    "{scene}_vh_clean_2.ply",
    "{scene}_vh_clean_2.0.010000.segs.json",
    "{scene}.aggregation.json",
    "{scene}.txt",
]

DERIVED_POSED_MIN_FILES = [
    "intrinsic.txt",
    "depth_intrinsic.txt",
]

def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, check=check)

def load_required_scenes(selection_json: Path) -> List[str]:
    """
    Supports both schemas:
    A) dict keyed by scene_id
    B) list of dicts with {"video_id":"scannet/sceneXXXX_YY", "frame_files":[...]}
    """
    if not selection_json.exists():
        return []
    data = json.loads(selection_json.read_text())
    scenes = set()

    if isinstance(data, dict):
        for k in data.keys():
            # could be "scene0050_00" or "scannet/scene0050_00"
            m = re.search(r"(scene\d{4}_\d{2})$", k)
            if m:
                scenes.add(m.group(1))
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            vid = (item.get("video_id") or "").replace("\\", "/")
            m = re.search(r"(scene\d{4}_\d{2})$", vid)
            if m:
                scenes.add(m.group(1))

    return sorted(scenes)

def check_scene_raw(scans_dir: Path, scene: str) -> Tuple[List[str], List[str]]:
    """Return (present, missing) raw filenames within scans/<scene>/."""
    scene_dir = scans_dir / scene
    missing, present = [], []
    for pat in RAW_SUFFIXES:
        fn = pat.format(scene=scene)
        p = scene_dir / fn
        (present if p.exists() else missing).append(fn)
    return present, missing

def check_scene_posed(posed_dir: Path, scene: str) -> Dict[str, object]:
    """
    Checks posed_images/<scene>/ for intrinsics and at least one jpg+pose.
    """
    scene_dir = posed_dir / scene
    out = {
        "exists": scene_dir.exists(),
        "missing": [],
        "num_jpg": 0,
        "num_pose_txt": 0,
    }
    if not scene_dir.exists():
        out["missing"] = ["(directory missing)"]
        return out

    # Minimum intrinsics files
    for f in DERIVED_POSED_MIN_FILES:
        if not (scene_dir / f).exists():
            out["missing"].append(f)

    jpgs = list(scene_dir.glob("*.jpg"))
    poses = [p for p in scene_dir.glob("*.txt") if p.name not in ("intrinsic.txt","depth_intrinsic.txt")]
    out["num_jpg"] = len(jpgs)
    out["num_pose_txt"] = len(poses)

    # Heuristic: you want some frames and some poses
    if len(jpgs) == 0:
        out["missing"].append("*.jpg frames (none found)")
    if len(poses) == 0:
        out["missing"].append("pose *.txt files (none found)")

    return out

def check_scene_views(scans_dir: Path, scene: str, expected_k: int = 32) -> Dict[str, object]:
    """
    Checks scans/<scene>/view_{k}.jpg existence.
    """
    scene_dir = scans_dir / scene
    missing = []
    present = 0
    for k in range(expected_k):
        p = scene_dir / f"view_{k}.jpg"
        if p.exists():
            present += 1
        else:
            missing.append(f"view_{k}.jpg")
    return {"present": present, "missing": missing}

def guess_download_script(scannet_root: Path) -> Optional[Path]:
    # You showed data/scannet/download-scannet.py exists
    p = scannet_root / "download-scannet.py"
    return p if p.exists() else None



def attempt_redownload_raw(
    download_py: Path,
    scannet_root: Path,
    scene: str,
    missing_raw: List[str],
    dry_run: bool,
) -> None:
    """
    Tailored to the official ScanNet download-scannet.py you pasted.

    That script:
      - Uses --id to download a single scan
      - Uses --type to download ONE file suffix (e.g. ".sens", "_vh_clean_2.ply", ".aggregation.json", ...)
      - Writes into: <out_dir>/scans/<scene>/<scene><suffix>

    It is interactive (TOS prompt), so we pipe a newline to auto-continue.
    """
    if not missing_raw:
        return

    # Map our missing filenames to the suffix expected by download-scannet.py (--type)
    # missing_raw entries look like: scene0000_02.sens, scene0000_02_vh_clean_2.ply, ...
    suffixes = []
    for fn in missing_raw:
        if not fn.startswith(scene):
            continue
        suffix = fn[len(scene):]  # keep leading '.' or '_' as required
        suffixes.append(suffix)

    # De-duplicate while preserving order
    seen = set()
    suffixes = [s for s in suffixes if not (s in seen or seen.add(s))]

    print(f"\n[re-download raw] {scene} missing {len(suffixes)} file types: {suffixes}")

    # Use --skip_existing to avoid re-downloading what is already present
    # Use printf "\n" | python ... to pass the TOS keypress prompt non-interactively
    for suf in suffixes:
        press = "\n\n" if suf == ".sens" else "\n"
        cmd = ["bash", "-lc", f'printf "{press}" | python "{download_py}" -o "{scannet_root}" --id "{scene}" --type "{suf}" --skip_existing']
        if dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue
        try:
            run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Failed downloading {scene}{suf}. You can try manually:")
            print(f'   printf "\\n" | python "{download_py}" -o "{scannet_root}" --id "{scene}" --type "{suf}" --skip_existing')
            # continue to next suffix rather than stopping

def regenerate_views_from_selection(
    selection_json: Path,
    scannet_root: Path,
    expected_k: int,
    dry_run: bool,
) -> None:
    """
    Creates scans/<scene>/view_{k}.jpg by symlinking/copying the selected
    posed_images frame_files from scannet_select_frames.json (your exact schema).
    """
    if not selection_json.exists():
        print("No selection JSON found; skipping view_{k}.jpg regeneration.")
        return

    data = json.loads(selection_json.read_text())
    scans = scannet_root / "scans"
    posed = scannet_root / "posed_images"

    def link_or_copy(src: Path, dst: Path):
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)

    # Normalize into mapping: scene -> list[Path]
    mapping: Dict[str, List[Path]] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            m = re.search(r"(scene\d{4}_\d{2})$", k)
            if not m:
                continue
            scene = m.group(1)
            if v and isinstance(v[0], int):
                mapping[scene] = [(posed/scene/f"{i:05d}.jpg") for i in v]
            elif v and isinstance(v[0], str):
                mapping[scene] = [Path(p) for p in v]
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            vid = (item.get("video_id") or "").replace("\\", "/")
            m = re.search(r"(scene\d{4}_\d{2})$", vid)
            if not m:
                continue
            scene = m.group(1)
            ff = item.get("frame_files") or []
            mapping[scene] = [Path(p) for p in ff]

    for scene, frames in mapping.items():
        dst_dir = scans / scene
        dst_dir.mkdir(parents=True, exist_ok=True)
        # create up to expected_k
        for k in range(min(expected_k, len(frames))):
            src = frames[k]
            dst = dst_dir / f"view_{k}.jpg"
            if dry_run:
                print(f"[DRY] link/copy {src} -> {dst}")
            else:
                if src.exists():
                    link_or_copy(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scannet_root", default="data/scannet", type=str)
    ap.add_argument("--selection_json", default="data/metadata/scannet_select_frames.json", type=str)
    ap.add_argument("--expected_views", default=32, type=int)
    ap.add_argument("--mode", choices=["audit", "repair_raw", "repair_views", "repair_all"], default="audit")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    scannet_root = Path(args.scannet_root)
    scans_dir = scannet_root / "scans"
    posed_dir = scannet_root / "posed_images"
    selection_json = Path(args.selection_json)

    required_scenes = load_required_scenes(selection_json)
    if not required_scenes:
        # Fallback: scenes that exist in scans/
        required_scenes = sorted([p.name for p in scans_dir.iterdir() if p.is_dir() and re.match(r"scene\d{4}_\d{2}", p.name)])
        print("⚠️  Could not infer scenes from selection JSON; falling back to scans/* directories.")

    # Audit
    print("\n=== ScanNet audit ===")
    print("scannet_root:", scannet_root.resolve())
    print("num_required_scenes:", len(required_scenes))
    print("expected_views_per_scene:", args.expected_views)
    print("selection_json:", selection_json)

    raw_missing_scenes = []
    posed_missing_scenes = []
    views_missing_scenes = []

    for scene in required_scenes:
        raw_present, raw_missing = check_scene_raw(scans_dir, scene)
        posed_stat = check_scene_posed(posed_dir, scene)
        views_stat = check_scene_views(scans_dir, scene, expected_k=args.expected_views)

        if raw_missing:
            raw_missing_scenes.append((scene, raw_missing))
        if posed_stat["missing"]:
            posed_missing_scenes.append((scene, posed_stat["missing"], posed_stat["num_jpg"], posed_stat["num_pose_txt"]))
        if views_stat["present"] < args.expected_views:
            views_missing_scenes.append((scene, views_stat["present"], len(views_stat["missing"])))

    print("\n--- RAW missing (re-download recommended) ---")
    if not raw_missing_scenes:
        print("None ✅")
    else:
        for scene, miss in raw_missing_scenes[:50]:
            print(f"{scene}: missing {len(miss)} raw files -> {miss}")
        if len(raw_missing_scenes) > 50:
            print(f"... ({len(raw_missing_scenes)-50} more)")

    print("\n--- POSED missing (regenerate from .sens recommended) ---")
    if not posed_missing_scenes:
        print("None ✅")
    else:
        for scene, miss, nj, nt in posed_missing_scenes[:50]:
            print(f"{scene}: missing={miss} | jpg={nj} pose_txt={nt}")
        if len(posed_missing_scenes) > 50:
            print(f"... ({len(posed_missing_scenes)-50} more)")

    print("\n--- VIEW_{k}.jpg missing (create from selection JSON) ---")
    if not views_missing_scenes:
        print("None ✅")
    else:
        for scene, present, missing_count in views_missing_scenes[:50]:
            print(f"{scene}: present={present} missing={missing_count}")
        if len(views_missing_scenes) > 50:
            print(f"... ({len(views_missing_scenes)-50} more)")

    # Repair actions
    if args.mode in ("repair_raw", "repair_all"):
        download_py = guess_download_script(scannet_root)
        if not download_py:
            print("\n❌ download-scannet.py not found at scannet_root; cannot auto re-download.")
        else:
            for scene, miss in raw_missing_scenes:
                attempt_redownload_raw(download_py, scannet_root, scene, miss, dry_run=args.dry_run)

    if args.mode in ("repair_views", "repair_all"):
        print("\n=== Regenerating view_{k}.jpg from selection JSON ===")
        regenerate_views_from_selection(selection_json, scannet_root, args.expected_views, dry_run=args.dry_run)

    print("\nDone.")

if __name__ == "__main__":
    main()
