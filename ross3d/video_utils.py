import os
import json
import torch
import pickle
import cv2
import numpy as np
from PIL import Image
from transformers.image_utils import to_numpy_array
from torchvision.transforms import ToTensor
import json
from tqdm import tqdm
import random
import copy
import tempfile
import warnings

from ross3d.utils import rank0_print


def euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from p1 to each row in p2."""
    p1 = np.reshape(p1, (1, 3))
    return np.sum((p2 - p1) ** 2, axis=1)


def _parse_selected_status(selected_status, n_points: int) -> np.ndarray:
    """Convert selected_status into a bool mask with length n_points."""
    if selected_status is None or len(selected_status) == 0:
        return np.zeros(n_points, dtype=bool)

    ss = np.asarray(selected_status)
    if ss.dtype == bool:
        if ss.shape[0] != n_points:
            raise ValueError(f"selected_status boolean mask must have length {n_points}")
        return ss.copy()

    mask = np.zeros(n_points, dtype=bool)
    idx = ss.astype(int).ravel()
    if idx.size > 0:
        if np.any(idx < 0) or np.any(idx >= n_points):
            raise ValueError("selected_status indices out of range")
        mask[idx] = True
    return mask


def farthest_view_sampling(
    k: int,
    candidates,
    seed: int,
    selected_status=None,
):
    """Greedy farthest-point sampling over candidate camera centers."""
    points = np.asarray(candidates, dtype=float)
    n_points = points.shape[0]
    if n_points == 0 or k <= 0:
        return []
    if k > n_points:
        k = n_points

    selected_mask = _parse_selected_status(selected_status, n_points)
    dist = np.full(n_points, np.inf, dtype=float)
    point_left_idx = np.arange(n_points, dtype=int)
    selected_points = []

    np.random.seed(seed)

    if np.any(selected_mask):
        selected_points = point_left_idx[selected_mask].tolist()
        point_left_idx = point_left_idx[~selected_mask]

        for idx in selected_points:
            d = euclidean_dist(points[idx], points[point_left_idx])
            dist[point_left_idx] = np.minimum(dist[point_left_idx], d)

        selected_index = selected_points[-1]
        start_iter = 0
    else:
        selected_index = int(np.random.randint(0, n_points))
        selected_points.append(selected_index)
        point_left_idx = point_left_idx[point_left_idx != selected_index]
        start_iter = 1

    for _ in range(start_iter, k):
        if point_left_idx.size == 0:
            break

        active_point = points[selected_index]
        d = euclidean_dist(active_point, points[point_left_idx])
        dist[point_left_idx] = np.minimum(dist[point_left_idx], d)

        farthest_pos = int(np.argmax(dist[point_left_idx]))
        selected_index = int(point_left_idx[farthest_pos])
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, farthest_pos)

    return selected_points


def _expand_indices_to_target_count(indices, target_count: int):
    """Expand sampled indices to target_count deterministically by repeating entries.

    This keeps FVS compatible with existing training assumptions that
    force-sampled videos yield a fixed number of frames (e.g. 32), even when
    a scene has fewer available frames.
    """
    if target_count <= 0:
        return []
    if len(indices) == 0:
        return []
    if len(indices) == target_count:
        return indices

    if len(indices) > target_count:
        return indices[:target_count]

    # len(indices) < target_count: deterministically repeat via linspace mapping
    pos = np.linspace(0, len(indices) - 1, target_count, dtype=int)
    return [indices[i] for i in pos]

def convert_from_uvd(u, v, d, intr, pose):
    # extr = np.linalg.inv(pose)
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]
    
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def unproject(intrinsics, poses, depths):
    """
        intrinsics: (V, 4, 4)
        poses: (V, 4, 4)
        depths: (V, H, W)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W) / 1000       # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)      # (V, H*W, 4)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords


class VideoProcessor:
    def __init__(
        self, 
        video_folder="data", 
        annotation_dir="data/embodiedscan/",
        voxel_size=None,
        min_xyz_range=None,
        max_xyz_range=None,
        frame_sampling_strategy='uniform',
        val_box_type='pred',
        fvs_cache_file='data/metadata/scannet_fvs_selected_frames.json',
        occupancy_root=None,
        coordinates_root=None,
        obj3d_json_path=None,
        occ_obj3d_warn=False,
    ):
        self.video_folder = video_folder
        self.voxel_size = voxel_size
        self.min_xyz_range = torch.tensor(min_xyz_range) if min_xyz_range is not None else None
        self.max_xyz_range = torch.tensor(max_xyz_range) if max_xyz_range is not None else None
        self.frame_sampling_strategy = frame_sampling_strategy
        self.fvs_cache_file = fvs_cache_file
        self.occupancy_root = occupancy_root
        self.coordinates_root = coordinates_root
        self.obj3d_json_path = obj3d_json_path
        self.occ_obj3d_warn = bool(occ_obj3d_warn)
        self.obj3d_scene_to_objects = self._load_obj3d_annotations()
        self.scene = {}
        print('============ frame sampling strategy: {} ============='.format(self.frame_sampling_strategy))

        for split in ["train", "val", "test"]:
            with open(os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl"), "rb") as f:
                data = pickle.load(f)["data_list"]
                for item in data:
                    # item["sample_idx"]: "scannet/scene0415_00"
                    if item["sample_idx"].startswith("scannet"):
                        self.scene[item["sample_idx"]] = item

        self.scan2obj = {}

        for split in ['train', 'val']:
            box_type = "gt" if split == "train" else val_box_type
            filename = os.path.join("data", "metadata", f"scannet_{split}_{box_type}_box.json")
            with open(filename) as f:
                data = json.load(f)
                self.scan2obj.update(data)


        if 'mc' in self.frame_sampling_strategy:
            sampling_file = "data/metadata/scannet_select_frames.json"
            self.mc_sampling_files = {}
            with open(sampling_file) as f:
                data = json.load(f)
                for dd in data:
                    self.mc_sampling_files[dd['video_id']] = dd

            with open('data/metadata/pcd_discrete_0.1.pkl', 'rb') as f:
                pc_data = pickle.load(f)
            self.pc_min = {}
            self.pc_max = {}
            for scene_id in pc_data:
                pc_points = pc_data[scene_id]
                min_xyz = [1000, 1000, 1000]
                max_xyz = [-1000, -1000, -1000]
                for data in pc_points:
                    min_xyz = [min(v1, v2) for v1, v2 in zip(min_xyz, data)]
                    max_xyz = [max(v1, v2) for v1, v2 in zip(max_xyz, data)]
                self.pc_min[scene_id] = torch.Tensor(min_xyz) / 10
                self.pc_max[scene_id] = torch.Tensor(max_xyz) / 10

        if 'fvs' in self.frame_sampling_strategy:
            camera_center_file = "data/metadata/scannet_camera_centers.json"
            self.fvs_camera_centers = {}
            with open(camera_center_file) as f:
                data = json.load(f)
                for dd in data:
                    self.fvs_camera_centers[dd["video_id"]] = dd["camera_centers"]

            self.fvs_cache = {}
            if self.fvs_cache_file and os.path.exists(self.fvs_cache_file):
                try:
                    with open(self.fvs_cache_file) as f:
                        self.fvs_cache = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"[Warning] Failed to parse FVS cache JSON: {self.fvs_cache_file}. "
                        "Starting with empty cache."
                    )
                    self.fvs_cache = {}

    def _save_fvs_cache(self):
        if not self.fvs_cache_file:
            return
        cache_dir = os.path.dirname(self.fvs_cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="fvs_cache_", suffix=".json", dir=cache_dir or None)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.fvs_cache, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.fvs_cache_file)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _validate_fvs_cache_entry(self, cache_entry, frame_files, num_frames_to_sample):
        if cache_entry is None:
            return False
        if cache_entry.get("total_frames") != len(frame_files):
            return False

        cached_frame_files = cache_entry.get("frame_files")
        if not isinstance(cached_frame_files, list) or len(cached_frame_files) != len(frame_files):
            return False
        if cached_frame_files != frame_files:
            return False

        cached_indices = cache_entry.get("selected_indices")
        if not isinstance(cached_indices, list) or len(cached_indices) != num_frames_to_sample:
            return False
        if any((not isinstance(i, int)) for i in cached_indices):
            return False
        if any(i < 0 or i >= len(frame_files) for i in cached_indices):
            return False

        return True


    def sample_frame_files_mc(self, video_id: str, frames_upbound: int = 32, do_shift=False):
        mc_files = self.mc_sampling_files[video_id]
        frame_files = mc_files['frame_files'][:frames_upbound]
        voxel_nums = mc_files['voxel_nums'][:frames_upbound]

        ratio = 1.0
        if 'ratio95' in self.frame_sampling_strategy:
            ratio = 0.95
        elif 'ratio90' in self.frame_sampling_strategy:
            ratio = 0.9

        if ratio != 1.0:
            num_all_voxels = mc_files['num_all_voxels']
            out = []
            cc = 0
            for frame_file, voxel_num in zip(frame_files, voxel_nums):
                out.append(frame_file)
                cc += voxel_num
                if cc >= num_all_voxels * ratio:
                    break
            frame_files = out

        frame_files.sort(key=lambda file: int(file.split('/')[-1].split('.')[0]))
        # if do_shift:
        #     ori_len = len(frame_files)
        #     i = random.randint(0, len(frame_files)-1)
        #     frame_files = frame_files[-i:] + frame_files[:-i]
        #     assert len(frame_files) == ori_len
        return frame_files  


    def sample_frame_files(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        # video_file: scannet/scene00000_01

        # since the color images have the suffix .jpg
        # frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f)) and os.path.join(video_file, f).endswith(".jpg")]
        # frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        meta_info = self.scene[video_id]
        frame_files = [os.path.join(self.video_folder, img["img_path"]) for img in meta_info["images"]]

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        if force_sample:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = 10

        # For scannet, the RGB camera data is temporally synchronized with the depth sensor via hardware, providing synchronized depth and color capture at 30Hz
        # We follow embodiedscan by sampling one out of every ten images.
        avg_fps = 3
        
        total_frames = len(frame_files)
        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        frames = [frame_files[i] for i in sampled_indices]

        # BEV image
        scene_id = frame_files[0].split('/')[3]
        bev_file = f"{frame_files[0].split('/')[0]}/{frame_files[0].split('/')[1]}/scans/{scene_id}/view_0.jpg"
        # frames.append(bev_file)

        # frame_time = [i/3 for i in sampled_indices]
        # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

        # video_time = total_frames / avg_fps

        return frames, bev_file

    def sample_frame_files_fvs(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
        seed: int = 0,
    ):
        meta_info = self.scene[video_id]
        frame_files = [os.path.join(self.video_folder, img["img_path"]) for img in meta_info["images"]]

        if force_sample:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = 10

        camera_centers = self.fvs_camera_centers.get(video_id)
        if camera_centers is None:
            raise KeyError(f"Missing camera centers for {video_id} in scannet_camera_centers.json")
        if len(camera_centers) != len(frame_files):
            raise ValueError(
                f"camera center length mismatch for {video_id}: {len(camera_centers)} vs {len(frame_files)}"
            )

        cache_entry = self.fvs_cache.get(video_id)
        if self._validate_fvs_cache_entry(cache_entry, frame_files, num_frames_to_sample):
            sampled_indices = cache_entry["selected_indices"]
        else:
            sampled_indices = farthest_view_sampling(
                k=num_frames_to_sample,
                candidates=np.asarray(camera_centers, dtype=float),
                seed=seed,
            )
            sampled_indices = sorted(sampled_indices)
            sampled_indices = _expand_indices_to_target_count(sampled_indices, num_frames_to_sample)
            self.fvs_cache[video_id] = {
                "total_frames": len(frame_files),
                "selected_indices": sampled_indices,
                "selected_frame_files": [frame_files[i] for i in sampled_indices],
                "frame_files": frame_files,
            }
            self._save_fvs_cache()

        frames = [frame_files[i] for i in sampled_indices]

        scene_id = frame_files[0].split('/')[3]
        bev_file = f"{frame_files[0].split('/')[0]}/{frame_files[0].split('/')[1]}/scans/{scene_id}/view_0.jpg"

        return frames, bev_file

    def calculate_world_coords(
        self,
        video_id: str, 
        frame_files,
        do_normalize=False,
    ):
        meta_info = self.scene[video_id]
        scene_id = video_id.split('/')[-1]

        axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix']))
        depth_intrinsic = torch.from_numpy(np.array(meta_info["depth_cam2img"]))

        depths = []
        poses = []
 
        # Read and store the sampled frames
        for frame_path in frame_files:

            # depth image
            depth_path = frame_path.replace(".jpg", ".png")
            try:
                with Image.open(depth_path) as depth_img:
                    depth = np.array(depth_img).astype(np.int32)
                    depths.append(torch.from_numpy(depth))
            except:
                # print("Warning failed to load", depth_path, "using all-zero depth image")
                assert depth_path.split('/')[-1] == 'view_0.png'
                depth = np.zeros((480, 640)).astype(np.int32)
                depths.append(torch.from_numpy(depth))

            # pose
            pose_file = frame_path.replace("jpg", "txt")
            try:
                pose = np.loadtxt(pose_file)
                poses.append(torch.from_numpy(pose))
            except:
                # print("Warning failed to load", pose_file, "using all-zero pose")
                assert pose_file.split('/')[-1] == 'view_0.txt'
                pose = np.zeros((4, 4))
                poses.append(torch.from_numpy(pose))


        depths = torch.stack(depths)   # (V, H, W)
        poses = torch.stack([axis_align_matrix @ pose for pose in poses])     # (V, 4, 4)
        depth_intrinsic = depth_intrinsic.unsqueeze(0).repeat(len(frame_files), 1, 1)
        
        world_coords = unproject(depth_intrinsic.float(), poses.float(), depths.float())    # (V, H, W, 3)

        if do_normalize:
            world_coords = torch.maximum(world_coords, self.pc_min[scene_id].to(world_coords.device))
            world_coords = torch.minimum(world_coords, self.pc_max[scene_id].to(world_coords.device))
        
        return {
            "world_coords": world_coords,
        }

    def preprocess(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
    ):

        if 'mc' in self.frame_sampling_strategy:
            frame_files, bev_file = self.sample_frame_files_mc(
                video_id,
                frames_upbound=frames_upbound,
                do_shift=('shift' in self.frame_sampling_strategy),
            )
        elif 'fvs' in self.frame_sampling_strategy:
            frame_files, bev_file = self.sample_frame_files_fvs(
                video_id,
                force_sample=force_sample,
                frames_upbound=frames_upbound,
            )
        else:
            frame_files, bev_file = self.sample_frame_files(
                video_id,
                force_sample=force_sample,
                frames_upbound=frames_upbound,
            )

        video_dict = self.calculate_world_coords(
            video_id,
            frame_files,
            do_normalize=('norm' in self.frame_sampling_strategy),
        )
        world_coords = video_dict["world_coords"]
        V, H, W, _ = world_coords.shape
        
        # boundry
        world_coords_flat = world_coords.reshape(-1, 3)
        x_min, x_max = world_coords_flat[:, 0].min().item(), world_coords_flat[:, 0].max().item()
        y_min, y_max = world_coords_flat[:, 1].min().item(), world_coords_flat[:, 1].max().item()
        z_min, z_max = world_coords_flat[:, 2].min().item(), world_coords_flat[:, 2].max().item()
        boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        # x_max = min(world_coords_flat[:, 0].min().abs().item(), world_coords_flat[:, 0].max().item())
        # x_min = - x_max
        # y_max = min(world_coords_flat[:, 1].min().abs().item(), world_coords_flat[:, 1].max().item())
        # y_min = - y_max
        # z_min, z_max = world_coords_flat[:, 2].min().item(), world_coords_flat[:, 2].max().item()
        # boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        images = []
        for frame_file in frame_files:
            with Image.open(frame_file) as img:
                frame = img.convert("RGB")
                images.append(frame)

        crop_size = image_processor.crop_size["width"]
        if strategy == "resize":
            images = [frame.resize((crop_size, crop_size)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (384, 384), interpolation=cv2.INTER_NEAREST) for coords in world_coords] 
        elif strategy == "center_crop":
            new_height = crop_size
            new_width = int(W * (crop_size / H))
            images = [frame.resize((new_width, new_height)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (new_width, new_height), interpolation=cv2.INTER_NEAREST) for coords in world_coords]
            # Calculate the position and perform the center crop
            left = (new_width - crop_size) // 2
            right = left + crop_size
            top = (new_height - crop_size) // 2
            bottom = top + crop_size
            images = [frame.crop((left, top, right, bottom)) for frame in images]

            resized_coords = [coords[top:bottom, left:right, :] for coords in resized_coords]

        bev_size = 432
        with Image.open(bev_file) as img:
            bev_image = img.convert("RGB")

        if strategy == "resize":
            bev_image = bev_image.resize((bev_size, bev_size))

        elif strategy == "center_crop":
            new_height = bev_size
            new_width = int(W * (bev_size / H))
            bev_image = bev_image.resize((new_width, new_height))
            # Calculate the position and perform the center crop
            left = (new_width - bev_size) // 2
            right = left + bev_size
            top = (new_height - bev_size) // 2
            bottom = top + bev_size
            bev_image = bev_image.crop((left, top, right, bottom))
        
        # resized_coords_norm = []
        # for coords in resized_coords:
        #     new_coords = coords.copy()
        #     new_coords[...,0] = (new_coords[...,0] - x_min) / (x_max - x_min)
        #     new_coords[...,1] = (new_coords[...,1] - y_min) / (y_max - y_min)
        #     new_coords[...,2] = (new_coords[...,2] - z_min) / (z_max - z_min)
        #     resized_coords_norm.append(new_coords)

        # resized_coords_norm = torch.from_numpy(np.stack(resized_coords_norm))
        return {
            "images": images,
            "world_coords": torch.from_numpy(np.stack(resized_coords)),
            "video_size": len(images),
            "boundry": boundry,
            "objects": torch.tensor(self.scan2obj[video_id]),
            "bev_image": bev_image,
            "scene_id": video_id,
            "frame_ids": [os.path.splitext(os.path.basename(f))[0] for f in frame_files],
            "patch_occupancy": self._load_patch_occupancy_annotations(video_id, frame_files),
            "visible_bboxes": self._load_visible_bboxes_annotations(video_id, frame_files),
            "obj3d_annotations": self.obj3d_scene_to_objects.get(video_id, None),
            # "world_coords_norm": resized_coords_norm
        }



    def _load_patch_occupancy_annotations(self, video_id: str, frame_files):
        annotations = []
        if not self.occupancy_root:
            return [None for _ in frame_files]

        scene_name = video_id.split('/')[-1]
        for frame_file in frame_files:
            frame_id = os.path.splitext(os.path.basename(frame_file))[0]
            occ_path = os.path.join(
                self.occupancy_root,
                scene_name,
                "patch_occupancy",
                f"{frame_id}_patch_occupancy.json",
            )
            if os.path.exists(occ_path):
                with open(occ_path, 'r') as f:
                    annotations.append(json.load(f))
            else:
                annotations.append(None)
        return annotations

    def _load_visible_bboxes_annotations(self, video_id: str, frame_files):
        annotations = []
        if not self.coordinates_root:
            return [None for _ in frame_files]

        scene_name = video_id.split('/')[-1]
        for frame_file in frame_files:
            frame_id = os.path.splitext(os.path.basename(frame_file))[0]
            bbox_path = os.path.join(
                self.coordinates_root,
                scene_name,
                "visible_bboxes",
                f"{frame_id}_visible_bboxes.json",
            )
            if os.path.exists(bbox_path):
                with open(bbox_path, 'r') as f:
                    bbox_data = json.load(f)

                loaded_frame_id = str(bbox_data.get("frame_id", frame_id))
                if loaded_frame_id != frame_id:
                    rank0_print(
                        f"[Warning] visible_bboxes frame_id mismatch for {video_id}: "
                        f"expected {frame_id}, got {loaded_frame_id} ({bbox_path})"
                    )

                loaded_scene_id = str(bbox_data.get("scene_id", video_id))
                if loaded_scene_id != video_id:
                    rank0_print(
                        f"[Warning] visible_bboxes scene_id mismatch for {video_id}: "
                        f"expected {video_id}, got {loaded_scene_id} ({bbox_path})"
                    )

                annotations.append(bbox_data)
            else:
                annotations.append(None)

        assert len(annotations) == len(frame_files)
        return annotations

    def _load_obj3d_annotations(self):
        scene_to_objects = {}
        def _obj3d_warn(msg: str) -> None:
            if self.occ_obj3d_warn:
                warnings.warn(msg)

        if not self.obj3d_json_path:
            return scene_to_objects
        if not os.path.exists(self.obj3d_json_path):
            _obj3d_warn(f"[occ_obj3d_loss] obj3d_json_path not found: {self.obj3d_json_path}")
            return scene_to_objects

        try:
            with open(self.obj3d_json_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            _obj3d_warn(f"[occ_obj3d_loss] Failed to load obj3d json {self.obj3d_json_path}: {exc}")
            return scene_to_objects

        if not isinstance(data, list):
            _obj3d_warn(f"[occ_obj3d_loss] Invalid obj3d json format (expected list): {self.obj3d_json_path}")
            return scene_to_objects

        for scene_entry in data:
            if not isinstance(scene_entry, dict):
                _obj3d_warn("[occ_obj3d_loss] Invalid scene entry in obj3d json (expected dict).")
                continue
            scene_id = str(scene_entry.get("scene_id", "")).strip()
            if not scene_id:
                _obj3d_warn("[occ_obj3d_loss] Missing scene_id in obj3d json entry.")
                continue
            objects = scene_entry.get("objects", [])
            if not isinstance(objects, list):
                _obj3d_warn(f"[occ_obj3d_loss] scene={scene_id} has invalid objects field (expected list).")
                continue

            obj_map = scene_to_objects.setdefault(scene_id, {})
            for obj in objects:
                if not isinstance(obj, dict):
                    _obj3d_warn(f"[occ_obj3d_loss] scene={scene_id} has invalid object entry (expected dict).")
                    continue
                if "object_id" not in obj:
                    _obj3d_warn(f"[occ_obj3d_loss] scene={scene_id} object missing object_id.")
                    continue
                try:
                    object_id = int(obj["object_id"])
                except Exception:
                    _obj3d_warn(f"[occ_obj3d_loss] scene={scene_id} has non-integer object_id={obj.get('object_id')}.")
                    continue
                obj_map[object_id] = {
                    "bbox": obj.get("bbox", None),
                    "center": obj.get("center", None),
                    "object_label": obj.get("object_label", None),
                }

        return scene_to_objects

    def process_3d_video(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
    ):
        video_dict = self.preprocess(
            video_id,
            image_processor,
            force_sample,
            frames_upbound,
            strategy,
        )
        video_dict["images"] = image_processor.preprocess(video_dict["images"], return_tensors="pt")["pixel_values"]
        # video_dict["bev_image"] = image_processor.preprocess(video_dict["bev_image"], return_tensors="pt")["pixel_values"]
        video_dict["bev_image"] = ToTensor()(video_dict["bev_image"]).unsqueeze(0)
        return video_dict

    
    def discrete_point(self, xyz):
        xyz = torch.tensor(xyz)
        if self.min_xyz_range is not None:
            xyz = torch.maximum(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None:
            xyz = torch.minimum(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None:
            xyz = (xyz - self.min_xyz_range.to(xyz.device)) 
            
        xyz = xyz / self.voxel_size
        return xyz.round().int().tolist()
    

def merge_video_dict(video_dict_list):
    new_video_dict = {}
    new_video_dict['box_input'] = []
    assert len(video_dict_list) == 1
    new_video_dict['bev_image'] = torch.Tensor(video_dict_list[0]['bev_image'])
    for k in video_dict_list[0].keys():
        if k in ["world_coords", 'images', 'objects']:
            new_video_dict[k] = torch.stack([video_dict[k] for video_dict in video_dict_list])
        elif k in ['box_input']:
            for video_dict in video_dict_list:
                if video_dict[k] is not None:
                    new_video_dict[k].append(video_dict[k])
        elif k in ['patch_occupancy', 'visible_bboxes', 'frame_ids', 'scene_id', 'obj3d_annotations']:
            new_video_dict[k] = video_dict_list[0][k]

    new_video_dict['box_input'] = torch.Tensor(new_video_dict['box_input'])
    return new_video_dict
