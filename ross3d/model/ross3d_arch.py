#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, Dict
import torch.nn.functional as F

import math
import re
import os
import time
import torch
import torch._dynamo
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from einops import rearrange
from copy import deepcopy

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import (
    build_vision_projector,
    build_inv_projector,
)
from .pixel_decoder.builder import build_pixel_decoder

from ross3d.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ross3d.mm_utils import get_anyres_image_grid_shape
from ross3d.utils import rank0_print, rank_print
import random
import warnings


def rlog(msg):
    if os.getenv("ROSS3D_RLOG_DEBUG", "0") != "1":
        return
    if torch._dynamo.is_compiling():
        return
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[RANK {rank}] {msg}", flush=True)


def tshape(x):
    return "None" if x is None else tuple(x.shape)


if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _compile_disable = torch.compiler.disable
else:
    def _compile_disable(fn):
        return fn


class SimCLRStylePatchProjector(nn.Module):
    def __init__(self, d_model: int, d_proj: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
            nn.LayerNorm(d_proj),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Ross3DMetaModel:

    def __init__(self, config):
        super(Ross3DMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.mask_token = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
            self._init_mask_token_(self.mask_token)
            self._audit_special_param_finiteness("post_constructor_mask_token_register")

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
                self._audit_special_param_finiteness("post_constructor_image_newline_register")
            self._audit_special_param_finiteness("post_constructor_init")
            self._audit_special_param_aliases("post_constructor_init")

        d_model = getattr(config, "hidden_size", None)
        occ_geom_enabled = bool(getattr(config, "enable_occ_geom_loss", False))
        occ_temp_enabled = bool(getattr(config, "enable_occ_temp_loss", False))
        occ_obj3d_enabled = bool(getattr(config, "enable_occ_obj3d_loss", False))
        occ_aux_enabled = occ_geom_enabled or occ_temp_enabled or occ_obj3d_enabled
        if d_model is not None and occ_aux_enabled:
            d_proj = getattr(config, "occupancy_projector_dim", None)
            d_proj = d_model if (d_proj is None) else d_proj
            self.occupancy_patch_projector = SimCLRStylePatchProjector(d_model=d_model, d_proj=d_proj)
            self.occupancy_object_norm = nn.LayerNorm(d_proj)
            self.config.occupancy_projector_dim = d_proj

            self.occ_geom_patch_norm = nn.LayerNorm(d_proj)
            self.occ_geom_obj_query = nn.Sequential(
                nn.LayerNorm(d_proj),
                nn.Linear(d_proj, d_proj),
            )
            self.occ_geom_relation = nn.Sequential(
                nn.LayerNorm(3 * d_proj),
                nn.Linear(3 * d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, d_proj),
                nn.LayerNorm(d_proj),
            )
            self.occ_geom_mask_head = nn.Linear(d_proj, 1)
            self.occ_geom_center_head = nn.Linear(d_proj, 1)
            self.occ_geom_size_head = nn.Sequential(
                nn.LayerNorm(d_proj),
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 2),
            )
            self.occ_geom_vis_head = nn.Sequential(
                nn.LayerNorm(d_proj),
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 1),
            )
            self.occ_temp_projector = nn.Sequential(
                nn.LayerNorm(d_proj),
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, d_proj),
                nn.LayerNorm(d_proj),
            )
            self.occ_obj3d_head_shared = nn.Sequential(
                nn.LayerNorm(d_proj),
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
            )
            self.occ_obj3d_center_head = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 3),
            )
            self.occ_obj3d_size_head = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 3),
            )
        
        if hasattr(self.config, 'world_position_embedding_type'):
            from ross3d.model.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingMLP

            if "sample9" in self.config.world_position_embedding_type:
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                n_points = 5
            elif "minmax" in self.config.world_position_embedding_type:
                n_points = 2
            else:
                n_points = 1
        
            if "mlp" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingMLP(config.hidden_size, n_points=n_points)
            elif "sin3d" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size, n_points=n_points)
            # elif "slp" in self.config.world_position_embedding_type:
            #     self.world_position_embedding = PositionEmbeddingSine3DMLP(config.hidden_size, n_points=n_points)

        # self.mm_inv_projector = build_inv_projector(self.config)
        # self.mm_pixel_decoder = build_pixel_decoder(self.config)
        # # other necessary information for reconstruction
        # self.image_embed_len = math.ceil(
        #     (self.vision_tower.config.image_size // self.vision_tower.config.patch_size)
        #     / float(self.config.mm_spatial_pool_stride)) ** 2

        self._audit_special_param_finiteness("post_model_init_complete")
        self._audit_special_param_aliases("post_model_init_complete")

    def _init_mask_token_(self, mask_token: torch.Tensor, deterministic: bool = False) -> None:
        std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=torch.float32))
        with torch.no_grad():
            if deterministic:
                seed = int(os.getenv("ROSS3D_MASK_TOKEN_SANITIZE_SEED", "0"))
                cpu_gen = torch.Generator(device="cpu")
                cpu_gen.manual_seed(seed)
                init_cpu = torch.randn(mask_token.shape, generator=cpu_gen, dtype=torch.float32) * std
                mask_token.copy_(init_cpu.to(device=mask_token.device, dtype=mask_token.dtype))
            else:
                torch.nn.init.normal_(mask_token, mean=0.0, std=float(std.item()))

    def _audit_special_param_finiteness(self, stage: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        for pname in ["mask_token", "image_newline"]:
            p = getattr(self, pname, None)
            if p is None:
                continue
            t = p.detach()
            finite = torch.isfinite(t)
            finite_any = bool(finite.any().item())
            finite_all = bool(finite.all().item())
            nan_count = int(torch.isnan(t).sum().item())
            inf_count = int(torch.isinf(t).sum().item())
            if finite_any:
                vals = t[finite]
                tmin = float(vals.min().item())
                tmax = float(vals.max().item())
            else:
                tmin, tmax = None, None
            rank0_print(
                "[NAN_DEBUG][param_audit] "
                f"stage={stage} name={pname} shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                f"finite_all={finite_all} nan_count={nan_count} inf_count={inf_count} min={tmin} max={tmax}"
            )

    def _audit_special_param_aliases(self, stage: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        named = dict(self.named_parameters()) if hasattr(self, "named_parameters") else {}
        for pname in ["mask_token", "image_newline"]:
            p = getattr(self, pname, None)
            if p is None:
                continue
            pptr = int(p.data_ptr())
            try:
                psptr = int(p.untyped_storage().data_ptr())
            except Exception:
                psptr = None
            alias_name = None
            for oname, other in named.items():
                if other is p:
                    continue
                if int(other.data_ptr()) == pptr:
                    alias_name = oname
                    break
            rank0_print(
                "[NAN_DEBUG][param_alias] "
                f"stage={stage} name={pname} data_ptr={pptr} storage_ptr={psptr} "
                f"is_view={p._base is not None} alias_with={alias_name}"
            )

    def _sanitize_mask_token_if_nonfinite(self, stage: str) -> None:
        m = getattr(self, "mask_token", None)
        if m is None:
            return
        finite_all = bool(torch.isfinite(m.detach()).all().item())
        if finite_all:
            return
        self._audit_special_param_finiteness(f"sanitize_before/{stage}")
        self._init_mask_token_(m, deterministic=True)
        self._audit_special_param_finiteness(f"sanitize_after/{stage}")
        if not bool(getattr(self, "_mask_token_sanitize_logged_once", False)):
            rank0_print(f"[NAN_DEBUG][sanitize] stage={stage} action=reinit_mask_token deterministic=True")
            self._mask_token_sanitize_logged_once = True

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        # for pixel_decoder
        mm_pixel_decoder = model_args.mm_pixel_decoder
        pretrain_mm_inv_adapter = model_args.pretrain_mm_inv_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")
        self.config.mm_pixel_decoder = mm_pixel_decoder

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                # In some resume/load flows the modules were already built before this
                # method is called, so they may not be wrapped in single-item lists.
                vision_resampler = self.vision_resampler[0] if isinstance(self.vision_resampler, list) else self.vision_resampler
                vision_tower = self.vision_tower[0] if isinstance(self.vision_tower, list) else self.vision_tower
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.image_embed_len = math.ceil(
            (self.vision_tower.config.image_size // self.vision_tower.config.patch_size)
            / float(model_args.mm_spatial_pool_stride)) ** 2
        self.config.image_embed_len = self.image_embed_len
        self.config.image_mean = self.vision_tower.image_processor.image_mean
        self.config.image_std = self.vision_tower.image_processor.image_std
        self.config.decode_image_size = self.vision_tower.config.image_size // self.vision_tower.config.patch_size * 8  # 336 -> 192; 384 -> 216

        ### build CLIP-LLM projector
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        self._audit_special_param_finiteness("initialize_vision_modules.start")
        self._sanitize_mask_token_if_nonfinite("initialize_vision_modules.start")
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
            self.mask_token = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std,
                                           requires_grad=True)
            self._sanitize_mask_token_if_nonfinite("initialize_vision_modules.mm_projector_new")

            if "unpad" in mm_patch_merge_type:
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std,
                                                  requires_grad=True)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            self._audit_special_param_finiteness("initialize_vision_modules.before_checkpoint_load")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            mask_ckpt_keys = [k for k in mm_projector_weights.keys() if ("mask_token" in k) or ("image_newline" in k)]
            rank0_print(f"[NAN_DEBUG][ckpt_audit] path={pretrain_mm_mlp_adapter} has_mask_or_newline_keys={len(mask_ckpt_keys) > 0} keys={mask_ckpt_keys[:8]}")
            for k in mask_ckpt_keys[:4]:
                t = mm_projector_weights[k]
                if torch.is_tensor(t):
                    finite = torch.isfinite(t)
                    finite_any = bool(finite.any().item())
                    finite_all = bool(finite.all().item())
                    nan_count = int(torch.isnan(t).sum().item())
                    inf_count = int(torch.isinf(t).sum().item())
                    if finite_any:
                        vals = t[finite]
                        tmin = float(vals.min().item())
                        tmax = float(vals.max().item())
                    else:
                        tmin, tmax = None, None
                    rank0_print(f"[NAN_DEBUG][ckpt_audit] key={k} shape={tuple(t.shape)} dtype={t.dtype} finite_all={finite_all} nan_count={nan_count} inf_count={inf_count} min={tmin} max={tmax}")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            self._sanitize_mask_token_if_nonfinite("initialize_vision_modules.after_checkpoint_load")
            self._audit_special_param_finiteness("initialize_vision_modules.after_checkpoint_load")
            self._audit_special_param_aliases("initialize_vision_modules.after_checkpoint_load")

        self.config.ross_enable = False
        self._audit_special_param_finiteness("initialize_vision_modules.after_precision_cast_or_to")
        self._audit_special_param_aliases("initialize_vision_modules.after_precision_cast_or_to")
        self._sanitize_mask_token_if_nonfinite("initialize_vision_modules.before_return")
        self._audit_special_param_finiteness("initialize_vision_modules.before_return")
        if getattr(model_args, 'mm_pixel_decoder', False):
            self.config.ross_enable = True
            ### build pixel decoder
            self.mm_pixel_decoder = build_pixel_decoder(self.config)
            self.config.mm_inv_hidden_size = self.mm_pixel_decoder.latent_dim

            ### build LLM-CLIP projector
            self.config.use_mm_inv_proj = True
            self.config.mm_inv_projector_type = getattr(model_args, 'mm_inv_projector_type', 'linear')

            if getattr(self, 'mm_inv_projector', None) is None:
                self.mm_inv_projector = build_inv_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_inv_projector.parameters():
                    p.requires_grad = True

            if pretrain_mm_inv_adapter is not None:
                rank0_print(f"=> loading pretrain_mm_inv_adapter from {pretrain_mm_inv_adapter} ...")
                mm_inv_projector_weights = torch.load(pretrain_mm_inv_adapter, map_location='cpu')

                def get_w(weights, keyword):
                    new_weights = {}
                    for k, v in weights.items():
                        if keyword in k:
                            new_k = k.split(keyword + '.')[1]
                            new_weights[new_k] = v

                    return new_weights

                # interpolate positional embeddings if necessary
                old_pos_embed = mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed"]
                old_h = old_w = int(math.sqrt(old_pos_embed.shape[1]))
                cur_h = cur_w = int(math.sqrt(self.mm_inv_projector.net.pos_embed_view.shape[1]))
                if old_h != cur_h:
                    rank0_print(f"=> interpolated pos_embed from {old_h}x{old_w} to {cur_h}x{cur_w}")
                    old_pos_embed = rearrange(old_pos_embed, 'b (h w) c -> b c h w', h=old_h, w=old_w)
                    new_pos_embed = torch.nn.functional.interpolate(
                        old_pos_embed,
                        size=(cur_h, cur_w),
                        mode='bilinear',
                    )
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_view"] = rearrange(
                        new_pos_embed, 'b c h w -> b (h w) c',
                    )
                else:
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_view"] = (
                        mm_inv_projector_weights)["model.mm_inv_projector.net.pos_embed"]

                # for shared weights
                cur_h = cur_w = int(math.sqrt(self.mm_inv_projector.net.pos_embed_bev.shape[1]))
                if old_h != cur_h:
                    rank0_print(f"=> interpolated pos_embed from {old_h}x{old_w} to {cur_h}x{cur_w}")
                    # old_pos_embed = rearrange(old_pos_embed, 'b (h w) c -> b c h w', h=old_h, w=old_w)
                    new_pos_embed = torch.nn.functional.interpolate(
                        old_pos_embed,
                        size=(cur_h, cur_w),
                        mode='bilinear',
                    )
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_bev"] = rearrange(
                        new_pos_embed, 'b c h w -> b (h w) c',
                    )
                else:
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_bev"] = (
                        mm_inv_projector_weights)["model.mm_inv_projector.net.pos_embed"]

                # rename weights
                old_weights = deepcopy(mm_inv_projector_weights)
                for k, v in old_weights.items():
                    if "net.x_embedder." in k:
                        k_view = k.replace(".x_embedder.", ".x_embedder_view.")
                        k_bev = k.replace(".x_embedder.", ".x_embedder_bev.")
                        mm_inv_projector_weights[k_view] = v
                        mm_inv_projector_weights[k_bev] = v
                    if "net.z_embedder." in k:
                        k_view = k.replace(".z_embedder.", ".z_embedder_view.")
                        k_bev = k.replace(".z_embedder.", ".z_embedder_bev.")
                        mm_inv_projector_weights[k_view] = v
                        mm_inv_projector_weights[k_bev] = v

                msg = self.mm_inv_projector.load_state_dict(get_w(mm_inv_projector_weights, 'mm_inv_projector'), strict=False)
                print(msg)

            self.config.ross_multi_task = False
            if getattr(model_args, 'ross_multi_task', False):
                self.config.ross_multi_task = True
                # share weights
                

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class Ross3DMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def _occ_match_last_dim(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        cur_dim = int(x.shape[-1])
        if cur_dim == int(target_dim):
            return x
        if cur_dim > int(target_dim):
            return x[..., :target_dim]
        pad_shape = list(x.shape[:-1]) + [int(target_dim - cur_dim)]
        pad = torch.zeros(*pad_shape, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    @staticmethod
    @torch._dynamo.disable
    def _sample_posterior_latents(posterior):
        """Run posterior sampling in eager mode so Dynamo skips tracing randn."""
        return posterior.sample()

    @staticmethod
    @torch._dynamo.disable
    def _pack_vae_latents(z_q):
        """Run latent packing in eager mode to avoid Dynamo SymInt/einops issues."""
        if z_q.shape[-1] % 2 == 1:
            z_q = nn.functional.interpolate(z_q, size=(z_q.shape[-2] + 1, z_q.shape[-1] + 1), mode='bilinear')
        # group each (2x2) window
        z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
        z_q = rearrange(z_q, 'b c h w p1 p2 -> b (c p1 p2) h w').contiguous()
        return z_q

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=None):
        if stride is None:
            stride = self.config.mm_spatial_pool_stride
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        # bchw
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        # if self.config.mm_spatial_pool_mode == "average":
        #     image_feature = nn.functional.avg_pool2d(image_feature, stride)
        # elif self.config.mm_spatial_pool_mode == "max":
        #     image_feature = nn.functional.max_pool2d(image_feature, stride)
        # elif self.config.mm_spatial_pool_mode == "bilinear":
        #     height, width = image_feature.shape[2:]
        #     scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        #     image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        # else:
        #     raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        height, width = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature


    def _world_coords_pool_params(self, world_coords, stride):
        num_patches = self.get_vision_tower().num_patches_per_side
        target_grid = math.ceil(num_patches / float(stride))
        _, height, width, _ = world_coords.size()
        patch_h = max(1, height // target_grid)
        patch_w = max(1, width // target_grid)
        crop_h = patch_h * target_grid
        crop_w = patch_w * target_grid
        return target_grid, patch_h, patch_w, crop_h, crop_w

    def average_coordinate_in_patch(self, world_coords, stride=None):
        if stride is None:
            stride = self.config.mm_spatial_pool_stride

        V, H, W, D = world_coords.size() # D = 3
        target_grid, patch_h, patch_w, crop_h, crop_w = self._world_coords_pool_params(world_coords, stride)

        world_coords = world_coords.view(V, H, W, D)[:, :crop_h, :crop_w, :]
        world_coords = world_coords.permute(0, 3, 1, 2)
        world_coords_avg = torch.nn.functional.avg_pool2d(
            world_coords,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)

        return world_coords_avg


    def minmax_coordinate_in_patch(self, world_coords, stride=None):
        if stride is None:
            stride = self.config.mm_spatial_pool_stride

        V, H, W, D = world_coords.size() # D = 3

        target_grid, patch_h, patch_w, crop_h, crop_w = self._world_coords_pool_params(world_coords, stride)
        world_coords = world_coords.view(V, H, W, D)[:, :crop_h, :crop_w, :]
        world_coords = world_coords.permute(0, 3, 1, 2)

        world_coords_max = torch.nn.functional.max_pool2d(
            world_coords,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        world_coords_max = world_coords_max.permute(0, 2, 3, 1)

        world_coords_min = -torch.nn.functional.max_pool2d(
            -world_coords,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        world_coords_min = world_coords_min.permute(0, 2, 3, 1)
        world_coords = torch.stack([world_coords_min, world_coords_max], dim=3)

        return world_coords


    def sample_n_points(self, world_coords, n_points=9, stride=None):
        if stride is None:
            stride = self.config.mm_spatial_pool_stride

        V, H, W, D = world_coords.size() # D = 3
        target_grid, patch_h, patch_w, crop_h, crop_w = self._world_coords_pool_params(world_coords, stride)
        world_coords = world_coords.view(V, H, W, D)[:, :crop_h, :crop_w, :]
        world_coords = world_coords.view(
            V,
            target_grid,
            patch_h,
            target_grid,
            patch_w,
            3,
        ).permute(0, 1, 3, 2, 4, 5)
        y_positions = ((torch.arange(3, device=world_coords.device) + 0.5) * (patch_h / 3) - 0.5).round().long()
        x_positions = ((torch.arange(3, device=world_coords.device) + 0.5) * (patch_w / 3) - 0.5).round().long()
        y_positions = torch.clamp(y_positions, 0, patch_h - 1)
        x_positions = torch.clamp(x_positions, 0, patch_w - 1)
        sampled = world_coords[:, :, :, y_positions][:, :, :, :, x_positions]
        sampled = sampled.reshape(V, target_grid, target_grid, 9, 3)
        if n_points == 9:
            world_coords_sample = sampled
        elif n_points == 5:
            world_coords_sample = sampled[:, :, :, [0, 2, 4, 6, 8], :]
        elif n_points == 1:
            world_coords_sample = sampled[:, :, :, 4, :].reshape(V, target_grid, target_grid, 3)
        else:
            raise NotImplementedError
        
        return world_coords_sample


    def discrete_coords(self, world_coords, xyz_min):

        # V, H, W, D = world_coords.size() # D = 3
        # world_coords_discrete = (world_coords.view(-1, 3) - xyz_min.view(1, 3)) / self.config.voxel_size

        min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(self.config.max_xyz_range).to(world_coords.device)

        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / self.config.voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach()


    @_compile_disable
    def _mm_projector_eager(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_model().mm_projector(x)

    def _run_mm_projector_debuggable(self, image_features: torch.Tensor) -> torch.Tensor:
        model = self.get_model()
        use_fp32 = os.getenv("ROSS3D_MM_PROJECTOR_FP32_DEBUG", "0") == "1"
        internal_debug = os.getenv("ROSS3D_MM_PROJECTOR_INTERNAL_DEBUG", "0") == "1"
        disable_compile = os.getenv("ROSS3D_DISABLE_MM_PROJECTOR_COMPILE", "0") == "1"

        self._maybe_install_mm_backward_hooks()
        projector_in = image_features.float() if use_fp32 else image_features
        self._debug_tensor_finite_stats("projector_internal.forward/input", projector_in)
        self._retain_and_track_grad("projector_input", projector_in)
        self._retain_and_track_grad("mm_projector_input", projector_in)

        if internal_debug and isinstance(model.mm_projector, nn.Sequential) and len(model.mm_projector) >= 3:
            l0 = model.mm_projector[0](projector_in)
            self._debug_tensor_finite_stats("projector_internal.forward/l0_out", l0)
            self._retain_and_track_grad("projector_l0_out", l0)

            act = model.mm_projector[1](l0)
            self._debug_tensor_finite_stats("projector_internal.forward/act_out", act)
            self._retain_and_track_grad("projector_act_out", act)

            l2 = model.mm_projector[2](act)
            self._debug_tensor_finite_stats("projector_internal.forward/l2_out", l2)
            self._retain_and_track_grad("projector_l2_out", l2)
            projector_out = l2
        else:
            if disable_compile:
                projector_out = self._mm_projector_eager(projector_in)
            else:
                projector_out = model.mm_projector(projector_in)

        self._debug_tensor_finite_stats("projector_internal.forward/output", projector_out)
        self._retain_and_track_grad("projector_output", projector_out)
        self._retain_and_track_grad("mm_projector_output", projector_out)

        if use_fp32:
            return projector_out.to(model.embed_tokens.weight.dtype)
        return projector_out

    def encode_images(self, images, world_coords=None):
        nan_dbg_enabled = os.getenv("ROSS3D_NAN_DEBUG", "0") == "1"

        def _nan_debug_stage(tag: str, tensor: Optional[torch.Tensor]):
            if not nan_dbg_enabled or tensor is None or (not torch.is_tensor(tensor)):
                return
            if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return
            count = int(getattr(self.config, "_nan_debug_encode_count", 0))
            max_logs = int(os.getenv("ROSS3D_NAN_DEBUG_MAX", "64"))
            if count >= max_logs:
                return
            setattr(self.config, "_nan_debug_encode_count", count + 1)
            t = tensor.detach()
            finite_mask = torch.isfinite(t)
            finite_any = bool(finite_mask.any().item())
            nan_count = int(torch.isnan(t).sum().item())
            inf_count = int(torch.isinf(t).sum().item())
            if finite_any:
                vals = t[finite_mask]
                minmax_msg = f"min={float(vals.min().item()):.6e} max={float(vals.max().item()):.6e}"
            else:
                minmax_msg = "min=NA max=NA"
            rank0_print(
                "[NAN_DEBUG][encode_images] "
                f"stage={tag} shape={tuple(t.shape)} dtype={t.dtype} "
                f"nan_count={nan_count} inf_count={inf_count} finite_any={finite_any} {minmax_msg}"
            )

        _nan_debug_stage("input_images", images)
        image_features = self.get_model().get_vision_tower()(images)
        _nan_debug_stage("vision_tower_out", image_features)

        if hasattr(self.get_model(), "vision_resampler") and (self.get_model().vision_resampler is not None):
            resampler_out = self.get_model().vision_resampler(image_features, images=images)
            _nan_debug_stage("vision_resampler_out", resampler_out)

        image_features = self._run_mm_projector_debuggable(image_features)
        self._trace_tensor_state("encode_images.mm_projector_out", image_features)
        _nan_debug_stage("mm_projector_out", image_features)

        return image_features


    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self._run_mm_projector_debuggable(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat, cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)

        return all_videos_or_images_features, all_faster_video_features


    def add_token_per_grid(self, image_feature, image_sizes=None):
        # infer per-frame (h, w) from image_sizes when available; otherwise fall back to square
        num_frames = image_feature.shape[0]
        num_tokens = image_feature.shape[1]
        feature_dim = image_feature.shape[-1]
        if (image_sizes is not None) and (len(image_sizes) > 0):
            # image_sizes: list of (H, W); both are pixel sizes, while num_tokens is patch count
            # use aspect ratio to recover patch grid (nearest integers that multiply to num_tokens)
            H, W = image_sizes[0]
            aspect = max(H, 1e-6) / max(W, 1e-6)
            resize_h = max(1, round((num_tokens ** 0.5) * (aspect ** 0.5)))
            resize_w = max(1, math.ceil(num_tokens / resize_h))
        else:
            resize_h = int(math.sqrt(num_tokens))
            resize_w = resize_h
        old_image_feature = image_feature.clone().detach()
        boi_ids = [None for _ in range(num_frames)]
        eoi_ids = [None for _ in range(num_frames)]

        # [32, 196, 3584] --> [32, 1, 14, 14, 3584]
        image_feature = image_feature.view(num_frames, 1, resize_h, resize_w, -1)
        # [32, 1, 14, 14, 3584] --> [3584, 32, 14, 1, 14]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        # [3584, 32, 14, 1, 14] --> [3584, 448, 1, 14] --> [3584, 448, 14]
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        # [3584, 448, 15]
        newline_insert_enabled = os.getenv("ROSS3D_DISABLE_IMAGE_NEWLINE_INSERT", "0") != "1"
        if newline_insert_enabled:
            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        if newline_insert_enabled:
            newline_ids = [resize_w + i * (resize_w + 1) for i in range(num_frames * num_tokens // resize_w)]
        else:
            newline_ids = []

        for image_id in range(old_image_feature.shape[0]):
            frame_stride = resize_h * (resize_w + (1 if newline_insert_enabled else 0))
            frame_base = image_id * frame_stride
            expected_boi = frame_base
            expected_eoi = frame_base + (resize_h - 1) * (resize_w + (1 if newline_insert_enabled else 0)) + (resize_w - 1)
            boi_ids[image_id] = int(expected_boi)
            eoi_ids[image_id] = int(expected_eoi)
            # skip strict equality when grids are non-square; rely on position math instead
            if resize_h == resize_w:
                boi_idx = boi_ids[image_id]
                eoi_idx = eoi_ids[image_id]
                in_bounds_boi = 0 <= boi_idx < image_feature.shape[0]
                in_bounds_eoi = 0 <= eoi_idx < image_feature.shape[0]
                if (not in_bounds_boi) or (not in_bounds_eoi):
                    raise RuntimeError(
                        "[prepare_mm][add_token_per_grid] index out of bounds "
                        f"image_id={image_id} boi_idx={boi_idx} eoi_idx={eoi_idx} "
                        f"packed_len={image_feature.shape[0]} resize_h={resize_h} resize_w={resize_w}"
                    )

                lhs_first = old_image_feature[image_id, 0]
                rhs_first = image_feature[boi_idx]
                lhs_last = old_image_feature[image_id, -1]
                rhs_last = image_feature[eoi_idx]
                atol = 1e-3 if lhs_first.dtype in (torch.float16, torch.bfloat16) else 1e-5
                rtol = 1e-3 if lhs_first.dtype in (torch.float16, torch.bfloat16) else 1e-5
                first_match = torch.isclose(lhs_first, rhs_first, atol=atol, rtol=rtol)
                last_match = torch.isclose(lhs_last, rhs_last, atol=atol, rtol=rtol)
                ok_first = bool(torch.all(first_match).item())
                ok_last = bool(torch.all(last_match).item())
                if (not ok_first) or (not ok_last):
                    def _tensor_finite_stats(x):
                        finite = torch.isfinite(x)
                        finite_all = bool(finite.all().item())
                        nan_count = int(torch.isnan(x).sum().item())
                        inf_count = int(torch.isinf(x).sum().item())
                        if bool(finite.any().item()):
                            max_abs = float(x[finite].abs().max().item())
                        else:
                            max_abs = float("nan")
                        return finite_all, nan_count, inf_count, max_abs

                    first_eq_count = int(first_match.sum().item())
                    last_eq_count = int(last_match.sum().item())
                    first_diff = (lhs_first - rhs_first).abs()
                    last_diff = (lhs_last - rhs_last).abs()
                    first_finite = torch.isfinite(first_diff)
                    last_finite = torch.isfinite(last_diff)
                    first_max_diff = float(first_diff[first_finite].max().item()) if bool(first_finite.any().item()) else float("nan")
                    last_max_diff = float(last_diff[last_finite].max().item()) if bool(last_finite.any().item()) else float("nan")
                    lhs_first_stats = _tensor_finite_stats(lhs_first)
                    rhs_first_stats = _tensor_finite_stats(rhs_first)
                    lhs_last_stats = _tensor_finite_stats(lhs_last)
                    rhs_last_stats = _tensor_finite_stats(rhs_last)
                    mapping_preview = []
                    preview_n = min(20, frame_stride)
                    for pidx in range(preview_n):
                        row = pidx // (resize_w + (1 if newline_insert_enabled else 0))
                        col = pidx % (resize_w + (1 if newline_insert_enabled else 0))
                        token_kind = "newline" if (newline_insert_enabled and col == resize_w) else "patch"
                        mapping_preview.append(f"{frame_base + pidx}:{token_kind}@({row},{col})")
                    patch_candidates = [0, 1, max(0, resize_w - 1), resize_w, max(0, num_tokens - 1)]
                    patch_diag = []
                    for patch_idx in patch_candidates:
                        if patch_idx < 0 or patch_idx >= num_tokens:
                            continue
                        prow = patch_idx // resize_w
                        pcol = patch_idx % resize_w
                        packed_idx = frame_base + prow * (resize_w + (1 if newline_insert_enabled else 0)) + pcol
                        if 0 <= packed_idx < image_feature.shape[0]:
                            cand = image_feature[packed_idx]
                            cand_match = torch.isclose(old_image_feature[image_id, patch_idx], cand, atol=atol, rtol=rtol)
                            cand_eq = int(cand_match.sum().item())
                            cand_diff = (old_image_feature[image_id, patch_idx] - cand).abs()
                            cand_fin = torch.isfinite(cand_diff)
                            cand_max = float(cand_diff[cand_fin].max().item()) if bool(cand_fin.any().item()) else float("nan")
                            lhs_stats = _tensor_finite_stats(old_image_feature[image_id, patch_idx])
                            rhs_stats = _tensor_finite_stats(cand)
                            patch_diag.append(
                                f"patch={patch_idx}->packed={packed_idx} eq={cand_eq}/{feature_dim} max_abs_diff={cand_max:.6e} "
                                f"lhs_finite={lhs_stats[0]} rhs_finite={rhs_stats[0]}"
                            )
                    if os.getenv("ROSS3D_RLOG_DEBUG", "0") == "1":
                        rlog(
                            "[prepare_mm][grid_assert] "
                            f"image_id={image_id} feature_dim={feature_dim} "
                            f"old_shape={tuple(old_image_feature.shape)} packed_shape={tuple(image_feature.shape)} "
                            f"boi_idx={boi_idx} eoi_idx={eoi_idx} "
                            f"expected_boi={expected_boi} expected_eoi={expected_eoi} "
                            f"in_bounds_boi={in_bounds_boi} in_bounds_eoi={in_bounds_eoi} "
                            f"first_eq={first_eq_count}/{feature_dim} last_eq={last_eq_count}/{feature_dim} "
                            f"first_max_abs_diff={first_max_diff:.6e} last_max_abs_diff={last_max_diff:.6e} "
                            f"mm_patch_merge_type={getattr(self.config, 'mm_patch_merge_type', 'NA')} "
                            f"mm_newline_position={getattr(self.config, 'mm_newline_position', 'NA')} "
                            f"num_frames={num_frames} num_tokens={num_tokens} resize_h={resize_h} resize_w={resize_w} "
                            f"newline_insert_enabled={newline_insert_enabled} "
                            f"image_sizes={image_sizes}"
                        )
                        rlog(
                            "[prepare_mm][grid_assert][finite] "
                            f"lhs_first={lhs_first_stats} rhs_first={rhs_first_stats} "
                            f"lhs_last={lhs_last_stats} rhs_last={rhs_last_stats}"
                        )
                        rlog("[prepare_mm][grid_assert][mapping_preview] " + " ".join(mapping_preview))
                        if len(patch_diag) > 0:
                            rlog("[prepare_mm][grid_assert][patch_diag] " + " | ".join(patch_diag))
                    raise RuntimeError(
                        "[prepare_mm][add_token_per_grid] BOI/EOI alignment mismatch "
                        f"image_id={image_id} first_eq={first_eq_count}/{feature_dim} "
                        f"last_eq={last_eq_count}/{feature_dim} "
                        f"first_max_abs_diff={first_max_diff:.6e} last_max_abs_diff={last_max_diff:.6e}"
                    )

        return image_feature, boi_ids, eoi_ids, old_image_feature, newline_ids

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        if os.getenv("ROSS3D_DISABLE_IMAGE_NEWLINE_INSERT", "0") != "1":
            image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def _nan_debug_rank0_enabled(self) -> bool:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return False
        return True

    def _debug_tensor_finite_stats(self, tag: str, tensor: Optional[torch.Tensor]) -> None:
        if (not self._nan_debug_rank0_enabled()) or tensor is None or (not torch.is_tensor(tensor)):
            return
        t = tensor.detach()
        finite = torch.isfinite(t)
        has_finite = bool(finite.any().item())
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        if has_finite:
            vals = t[finite]
            min_v = float(vals.min().item())
            max_v = float(vals.max().item())
        else:
            min_v, max_v = None, None
        finite_all = bool(finite.all().item())
        rank0_print(
            f"[NAN_DEBUG][{tag}] shape={tuple(t.shape)} dtype={t.dtype} "
            f"finite_all={finite_all} finite_any={has_finite} "
            f"nan_count={nan_count} inf_count={inf_count} min={min_v} max={max_v}"
        )

    def _trace_tensor_state(self, tag: str, tensor: Optional[torch.Tensor]) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if (not self._nan_debug_rank0_enabled()) or tensor is None or (not torch.is_tensor(tensor)):
            return
        t = tensor.detach()
        finite = torch.isfinite(t)
        finite_any = bool(finite.any().item())
        finite_all = bool(finite.all().item())
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        if finite_any:
            vals = t[finite]
            tmin = float(vals.min().item())
            tmax = float(vals.max().item())
        else:
            tmin, tmax = None, None
        alias = None
        try:
            alias = int(t.untyped_storage().data_ptr())
        except Exception:
            alias = None
        rank0_print(
            f"[NAN_DEBUG][tensor_trace] tag={tag} shape={tuple(t.shape)} dtype={t.dtype} "
            f"finite_all={finite_all} finite_any={finite_any} nan_count={nan_count} inf_count={inf_count} "
            f"min={tmin} max={tmax} data_ptr={int(t.data_ptr())} storage_ptr={alias} "
            f"is_view={t._base is not None} stride={tuple(t.stride())}"
        )

    def _nan_debug_validate_layout_snapshot(self, boundary: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        snap = getattr(self, "_nan_debug_layout_snapshot", None)
        if not snap:
            return
        old_feat = snap.get("old_image_feature")
        packed = snap.get("cur_new_input_embeds")
        boi_ids = snap.get("boi_ids")
        eoi_ids = snap.get("eoi_ids")
        newline_ids = snap.get("newline_ids")
        newline_ref = snap.get("image_newline")
        if old_feat is None or packed is None or boi_ids is None or eoi_ids is None:
            return
        try:
            fp32_layout_tolerant = os.getenv("ROSS3D_MM_BRANCH_FP32_DEBUG", "0") == "1"
            if fp32_layout_tolerant:
                atol = float(os.getenv("ROSS3D_MM_LAYOUT_ATOL", "1e-2"))
                rtol = float(os.getenv("ROSS3D_MM_LAYOUT_RTOL", "1e-2"))
                first_match = torch.isclose(old_feat[:, 0].float(), packed[boi_ids].float(), atol=atol, rtol=rtol).all(dim=1)
                last_match = torch.isclose(old_feat[:, -1].float(), packed[eoi_ids].float(), atol=atol, rtol=rtol).all(dim=1)
            else:
                first_match = torch.all(old_feat[:, 0] == packed[boi_ids], dim=1)
                last_match = torch.all(old_feat[:, -1] == packed[eoi_ids], dim=1)
            ok_first = bool(torch.all(first_match).item())
            ok_last = bool(torch.all(last_match).item())
            ok_newline = True
            if newline_ids is not None and newline_ref is not None and len(newline_ids) > 0:
                if fp32_layout_tolerant:
                    newline_match = torch.isclose(
                        newline_ref.unsqueeze(0).repeat(len(newline_ids), 1).float(),
                        packed[newline_ids].float(),
                        atol=atol,
                        rtol=rtol,
                    ).all(dim=1)
                else:
                    newline_match = torch.all(newline_ref.unsqueeze(0).repeat(len(newline_ids), 1) == packed[newline_ids], dim=1)
                ok_newline = bool(torch.all(newline_match).item())
            rank0_print(
                f"[NAN_DEBUG][layout_snapshot] boundary={boundary} ok_first={ok_first} ok_last={ok_last} ok_newline={ok_newline}"
            )
            if os.getenv("ROSS3D_NAN_FAIL_FAST", "0") == "1" and (not ok_first or not ok_last or not ok_newline):
                raise RuntimeError(
                    f"[NAN_DEBUG] post-backward layout invalid at {boundary}: "
                    f"ok_first={ok_first} ok_last={ok_last} ok_newline={ok_newline}"
                )
        except Exception as e:
            rank0_print(f"[NAN_DEBUG][layout_snapshot] boundary={boundary} validate_error={e}")

    def _check_tensor_finite_or_raise(self, tag: str, tensor: Optional[torch.Tensor]) -> None:
        if os.getenv("ROSS3D_NAN_FAIL_FAST", "0") != "1":
            return
        if tensor is None or (not torch.is_tensor(tensor)):
            return
        if os.getenv("ROSS3D_SKIP_MASK_TOKEN_NAN_CHECK", "0") == "1":
            mask_token = getattr(self.get_model(), "mask_token", None)
            if torch.is_tensor(mask_token) and int(tensor.data_ptr()) == int(mask_token.data_ptr()):
                if self._nan_debug_rank0_enabled():
                    rank0_print(f"[NAN_DEBUG][{tag}] skip_mask_token_nan_check=True")
                return
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(
                f"[NAN_DEBUG][{tag}] non-finite tensor detected "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    def _log_alias_info(self, tag: str, tensor: Optional[torch.Tensor], ref: Optional[torch.Tensor], ref_name: str) -> None:
        if (not self._nan_debug_rank0_enabled()) or tensor is None or ref is None:
            return
        if (not torch.is_tensor(tensor)) or (not torch.is_tensor(ref)):
            return
        same_ptr = int(tensor.data_ptr()) == int(ref.data_ptr())
        rank0_print(
            f"[NAN_DEBUG][{tag}][ALIAS] tensor_ptr={int(tensor.data_ptr())} "
            f"ref={ref_name} ref_ptr={int(ref.data_ptr())} same_ptr={same_ptr} "
            f"tensor_base={tensor._base is not None}"
        )
        if os.getenv("ROSS3D_STRICT_ALIAS_CHECK", "0") == "1" and same_ptr:
            raise RuntimeError(f"[NAN_DEBUG][{tag}] alias detected with {ref_name}")

    def _clone_mm_insert_if_debug(self, tensor: torch.Tensor) -> torch.Tensor:
        if os.getenv("ROSS3D_CLONE_MM_INSERTS_DEBUG", "0") == "1":
            return tensor.clone()
        return tensor

    def _retain_and_track_grad(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if os.getenv("ROSS3D_NAN_RETAIN_FIRST_BATCH_ONLY", "1") == "1":
            if not bool(getattr(self, "_nan_debug_track_this_batch", False)):
                return
        if tensor is None or (not torch.is_tensor(tensor)) or (not tensor.requires_grad):
            return
        tensor.retain_grad()
        store = getattr(self, "_nan_debug_retained_tensors", None)
        if store is None:
            store = {}
            setattr(self, "_nan_debug_retained_tensors", store)
        store[name] = tensor
        order = getattr(self, "_nan_debug_retained_order", None)
        if order is None:
            order = []
            setattr(self, "_nan_debug_retained_order", order)
        if name not in order:
            order.append(name)

    def _log_grad_stats_for_tensor(self, tag: str, tensor: Optional[torch.Tensor]) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if not self._nan_debug_rank0_enabled():
            return
        if tensor is None or (not torch.is_tensor(tensor)):
            rank0_print(f"[NAN_DEBUG][{tag}] grad_state=none")
            return
        grad = tensor.grad
        if grad is None:
            rank0_print(f"[NAN_DEBUG][{tag}] grad_state=none")
            return
        g = grad.detach()
        finite = torch.isfinite(g)
        finite_any = bool(finite.any().item())
        finite_all = bool(finite.all().item())
        nan_count = int(torch.isnan(g).sum().item())
        inf_count = int(torch.isinf(g).sum().item())
        if finite_any:
            vals = g[finite]
            gmin = float(vals.min().item())
            gmax = float(vals.max().item())
        else:
            gmin, gmax = None, None
        state = "finite" if finite_all else "nonfinite"
        rank0_print(
            f"[NAN_DEBUG][{tag}] grad_state={state} shape={tuple(g.shape)} dtype={g.dtype} "
            f"nan_count={nan_count} inf_count={inf_count} min={gmin} max={gmax}"
        )

    def _maybe_log_projector_internal_backward(self, boundary_tag: str) -> None:
        if os.getenv("ROSS3D_MM_PROJECTOR_INTERNAL_DEBUG", "0") != "1":
            return
        store = getattr(self, "_nan_debug_retained_tensors", {})
        for key in ["projector_input", "projector_l0_out", "projector_act_out", "projector_l2_out", "projector_output"]:
            self._log_grad_stats_for_tensor(f"projector_internal.backward/{boundary_tag}/{key}", store.get(key, None))

    def _maybe_log_multimodal_backward_chain(self, boundary_tag: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        chain = [
            "mm_projector_input",
            "encoded_image_features",
            "mm_projector_output",
            "image_features_after_split_idx0",
            "image_feat_after_mask_idx0",
            "image_feat_after_world_pe_idx0",
            "image_feature_after_add_token_per_grid_idx0",
            "cur_new_input_embeds_after_cat",
            "boi_slice",
            "eoi_slice",
            "newline_slice",
        ]
        store = getattr(self, "_nan_debug_retained_tensors", {})
        last_finite = None
        first_nonfinite = None
        for key in chain:
            t = store.get(key, None)
            self._log_grad_stats_for_tensor(f"mm_chain/{boundary_tag}/{key}", t)
            g = None if t is None else t.grad
            if g is None:
                continue
            finite_all = bool(torch.isfinite(g.detach()).all().item())
            if finite_all:
                last_finite = key
            elif first_nonfinite is None:
                first_nonfinite = key
        rank0_print(
            f"[NAN_DEBUG][mm_chain/{boundary_tag}] last_finite={last_finite} first_nonfinite={first_nonfinite}"
        )

    def _maybe_install_mm_backward_hooks(self) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if getattr(self, "_nan_debug_mm_backward_hooks_installed", False):
            return
        model = self.get_model()
        handles = []

        def _make_hook(name: str):
            def _hook(module, grad_input, grad_output):
                if not self._nan_debug_rank0_enabled():
                    return
                def _state(g):
                    if g is None or (not torch.is_tensor(g)):
                        return "none"
                    return "finite" if bool(torch.isfinite(g.detach()).all().item()) else "nonfinite"
                in_state = [_state(g) for g in grad_input]
                out_state = [_state(g) for g in grad_output]
                rank0_print(f"[NAN_DEBUG][full_bw_hook] module={name} grad_input={in_state} grad_output={out_state}")
            return _hook

        if hasattr(model, "mm_projector") and isinstance(model.mm_projector, nn.Sequential):
            if len(model.mm_projector) > 0 and isinstance(model.mm_projector[0], nn.Module):
                handles.append(model.mm_projector[0].register_full_backward_hook(_make_hook("mm_projector.0")))
            if len(model.mm_projector) > 2 and isinstance(model.mm_projector[2], nn.Module):
                handles.append(model.mm_projector[2].register_full_backward_hook(_make_hook("mm_projector.2")))

        if hasattr(model, "world_position_embedding") and isinstance(model.world_position_embedding, nn.Module):
            handles.append(model.world_position_embedding.register_full_backward_hook(_make_hook("world_position_embedding")))

        self._nan_debug_mm_backward_hook_handles = handles
        self._nan_debug_mm_backward_hooks_installed = True

    def _maybe_log_newline_packed_grad(self, boundary_tag: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        records = getattr(self, "_nan_debug_packed_grad_records", None)
        if not records:
            return
        for ridx, rec in enumerate(records[:4]):
            packed = rec.get("packed")
            self._log_grad_stats_for_tensor(f"newline_grad/{boundary_tag}/packed_idx{ridx}", packed)
            if packed is None or packed.grad is None:
                continue
            g = packed.grad.detach()
            for k in ["newline_ids", "boi_ids", "eoi_ids"]:
                idx = rec.get(k, None)
                if idx is None or len(idx) == 0:
                    rank0_print(f"[NAN_DEBUG][newline_grad/{boundary_tag}] {k}_state=none")
                    continue
                idx_t = torch.as_tensor(idx, device=g.device, dtype=torch.long)
                idx_t = idx_t[(idx_t >= 0) & (idx_t < g.shape[0])]
                if idx_t.numel() == 0:
                    rank0_print(f"[NAN_DEBUG][newline_grad/{boundary_tag}] {k}_state=empty")
                    continue
                part = g[idx_t]
                finite = torch.isfinite(part)
                finite_any = bool(finite.any().item())
                finite_all = bool(finite.all().item())
                nan_count = int(torch.isnan(part).sum().item())
                inf_count = int(torch.isinf(part).sum().item())
                if finite_any:
                    vals = part[finite]
                    pmin = float(vals.min().item())
                    pmax = float(vals.max().item())
                else:
                    pmin, pmax = None, None
                state = "finite" if finite_all else "nonfinite"
                rank0_print(
                    f"[NAN_DEBUG][newline_grad/{boundary_tag}] region={k} grad_state={state} "
                    f"shape={tuple(part.shape)} nan_count={nan_count} inf_count={inf_count} min={pmin} max={pmax}"
                )

            boi = rec.get("boi_ids", None)
            eoi = rec.get("eoi_ids", None)
            if boi is not None and eoi is not None and len(boi) == len(eoi) and len(boi) > 0:
                image_ranges = []
                for b, e in zip(boi, eoi):
                    b_i = int(b)
                    e_i = int(e)
                    if e_i >= b_i:
                        image_ranges.extend(range(b_i, e_i + 1))
                if len(image_ranges) > 0:
                    idx_t = torch.as_tensor(sorted(set(image_ranges)), device=g.device, dtype=torch.long)
                    idx_t = idx_t[(idx_t >= 0) & (idx_t < g.shape[0])]
                    if idx_t.numel() > 0:
                        part = g[idx_t]
                        finite = torch.isfinite(part)
                        finite_any = bool(finite.any().item())
                        finite_all = bool(finite.all().item())
                        nan_count = int(torch.isnan(part).sum().item())
                        inf_count = int(torch.isinf(part).sum().item())
                        if finite_any:
                            vals = part[finite]
                            pmin = float(vals.min().item())
                            pmax = float(vals.max().item())
                        else:
                            pmin, pmax = None, None
                        state = "finite" if finite_all else "nonfinite"
                        rank0_print(
                            f"[NAN_DEBUG][newline_grad/{boundary_tag}] region=image_token grad_state={state} "
                            f"shape={tuple(part.shape)} nan_count={nan_count} inf_count={inf_count} min={pmin} max={pmax}"
                        )

    def replace_with_mask_token(self, x, mask_ratio):
        # x: [num_frames, num_patches, embed_dim]
        num_frames, num_patches, embed_dim = x.shape
        len_keep = int(num_frames * (1 - mask_ratio))
        len_mask = num_frames - len_keep

        noise = torch.rand(num_frames, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([num_frames], device=x.device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)

        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        x_masked = torch.gather(x, dim=0, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, num_patches, embed_dim))

        # append mask tokens
        if self._nan_debug_rank0_enabled():
            ratio = float(len_mask) / max(float(num_frames), 1.0)
            rank0_print(
                f"[NAN_DEBUG][mask_replace] len_mask={len_mask} len_keep={len_keep} num_frames={num_frames} "
                f"mask_ratio={ratio:.6f} inserted={len_mask > 0}"
            )
        mask_tokens = self.get_model().mask_token.unsqueeze(0).repeat(len_mask, num_patches, 1)
        self._log_alias_info("replace_with_mask_token.mask_tokens", mask_tokens, self.get_model().mask_token, "mask_token")
        mask_tokens = self._clone_mm_insert_if_debug(mask_tokens)
        self._debug_tensor_finite_stats("replace_with_mask_token.mask_tokens", mask_tokens)
        x_ = torch.cat([x_masked, mask_tokens], dim=0)
        x_ = torch.gather(x_, dim=0, index=ids_restore.unsqueeze(-1).unsqueeze(-1).repeat(1, num_patches, embed_dim))  # unshuffle
        self._debug_tensor_finite_stats("replace_with_mask_token.output", x_)
        if self._nan_debug_rank0_enabled() and x_.numel() > 0 and x.shape == x_.shape:
            changed = bool((x_ - x).abs().max().item() > 0.0)
            rank0_print(f"[NAN_DEBUG][mask_replace] output_changed={changed}")
        self._check_tensor_finite_or_raise("replace_with_mask_token.output", x_)

        return x_, mask


    @_compile_disable
    def _prepare_inputs_labels_for_multimodal_eager(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
        video_dict=None,
        use_object_proposals: bool = False,
        replace_with_mask_token: bool = False,
    ):
        return self._prepare_inputs_labels_for_multimodal_impl(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            modalities=modalities,
            image_sizes=image_sizes,
            video_dict=video_dict,
            use_object_proposals=use_object_proposals,
            replace_with_mask_token=replace_with_mask_token,
        )

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
        video_dict=None,
        use_object_proposals: bool = False,
        replace_with_mask_token: bool = False,
    ):
        return self._prepare_inputs_labels_for_multimodal_eager(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            modalities=modalities,
            image_sizes=image_sizes,
            video_dict=video_dict,
            use_object_proposals=use_object_proposals,
            replace_with_mask_token=replace_with_mask_token,
        )

    def _prepare_inputs_labels_for_multimodal_impl(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images: List[torch.FloatTensor],
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
        use_object_proposals: bool = False,
        replace_with_mask_token: bool = False,
    ):
        mm_branch_fp32 = os.getenv("ROSS3D_MM_BRANCH_FP32_DEBUG", "0") == "1"
        base_model = self.get_model()
        if hasattr(base_model, "_sanitize_mask_token_if_nonfinite") and os.getenv("ROSS3D_SANITIZE_MASK_TOKEN_PRE_FORWARD_ONLY", "0") != "1":
            base_model._sanitize_mask_token_if_nonfinite("prepare_mm.begin")
        self._debug_tensor_finite_stats("prepare_mm.mask_token.begin", getattr(self.get_model(), "mask_token", None))
        self._debug_tensor_finite_stats("prepare_mm.image_newline.begin", getattr(self.get_model(), "image_newline", None))
        object_boxes = None
        if use_object_proposals:
            object_boxes = video_dict["objects"][0]
            object_boxes_center = object_boxes[:, :3]
            object_features = []
            obj_num = len(object_boxes)

            object_patch = []
            # ignore the batch dimension here
            world_coords = video_dict["world_coords"][0]

            for l in range(obj_num):
                box = object_boxes[l]
                min_xyz = box[:3] - box[3:] / 2
                max_xyz = box[:3] + box[3:] / 2
                
                if "patch27" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 14, 27, 14, 27, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(27 * 27 * 0.25)
                    object_patch.append(cur_object_patch)
                elif "patch14" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 27, 14, 27, 14, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(14 * 14 * 0.5)
                    object_patch.append(cur_object_patch)
                else:
                    raise NotImplementedError


        use_mrope_position_embedding = False
        use_sin3d_pe = False
        use_mlp_pe = False
        if hasattr(self.config, 'world_position_embedding_type') and past_key_values is None:
            B = input_ids.shape[0]
            world_coords = video_dict['world_coords']
            xyz_min = world_coords.view(B, -1, 3).min(dim=1)[0]

            if len(video_dict['box_input']):
                box_input = video_dict['box_input']     # [1, 3]
            else:
                box_input = None

            n_points = 1
            if 'avg' in self.config.world_position_embedding_type:
                world_coords = [self.average_coordinate_in_patch(coords) for coords in world_coords]
            elif "sample9" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=9) for coords in world_coords]
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=5) for coords in world_coords]
                n_points = 5
            elif "sample1" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=1) for coords in world_coords]
            elif "minmax" in self.config.world_position_embedding_type:
                world_coords = [self.minmax_coordinate_in_patch(coords) for coords in world_coords]
                n_points = 2

            if n_points > 1:
                if box_input is not None:
                    box_input = box_input[:, None, :].repeat(1, n_points, 1)
                if object_boxes is not None:
                    object_boxes_center = object_boxes_center[:, None, :].repeat(1, n_points, 1)

            if 'discrete' in self.config.world_position_embedding_type or use_mrope_position_embedding:
                world_coords_discrete = [self.discrete_coords(coords, xyz_min[i]) for i, coords in enumerate(world_coords)]
                if box_input is not None:
                    box_input = self.discrete_coords(box_input, None)
                if object_boxes is not None:
                    object_boxes_center = self.discrete_coords(object_boxes_center, None)

            if 'mrope' in self.config.world_position_embedding_type:
                use_mrope_position_embedding = True
            
            if "sin3d" in self.config.world_position_embedding_type:
                use_sin3d_pe = True
            
            if "mlp" in self.config.world_position_embedding_type:
                use_mlp_pe = True


        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        nan_dbg_enabled = os.getenv("ROSS3D_NAN_DEBUG", "0") == "1"
        # Keep per-step debug tensor state from leaking across training steps.
        if nan_dbg_enabled:
            setattr(self, "_nan_debug_packed_grad_records", [])
            setattr(self, "_nan_debug_layout_snapshot", None)
            setattr(self, "_nan_debug_retained_tensors", {})
            setattr(self, "_nan_debug_retained_order", [])
        else:
            if hasattr(self, "_nan_debug_packed_grad_records"):
                setattr(self, "_nan_debug_packed_grad_records", [])
            if hasattr(self, "_nan_debug_layout_snapshot"):
                setattr(self, "_nan_debug_layout_snapshot", None)
            if hasattr(self, "_nan_debug_retained_tensors"):
                setattr(self, "_nan_debug_retained_tensors", {})
            if hasattr(self, "_nan_debug_retained_order"):
                setattr(self, "_nan_debug_retained_order", [])

        def _nan_debug_should_log() -> bool:
            if not nan_dbg_enabled:
                return False
            if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return False
            count = int(getattr(self.config, "_nan_debug_log_count", 0))
            max_logs = int(os.getenv("ROSS3D_NAN_DEBUG_MAX", "64"))
            if count >= max_logs:
                return False
            setattr(self.config, "_nan_debug_log_count", count + 1)
            return True

        def _nan_debug_tensor(tag: str, tensor: Optional[torch.Tensor], batch_idx: int = -1):
            if (tensor is None) or (not torch.is_tensor(tensor)):
                return
            if not _nan_debug_should_log():
                return
            with torch.no_grad():
                t = tensor.detach()
                finite_mask = torch.isfinite(t)
                finite_all = bool(finite_mask.all().item())
                finite_any = bool(finite_mask.any().item())
                nan_count = int(torch.isnan(t).sum().item())
                inf_count = int(torch.isinf(t).sum().item())
                if finite_any:
                    finite_vals = t[finite_mask]
                    min_val = float(finite_vals.min().item())
                    max_val = float(finite_vals.max().item())
                    minmax_msg = f"min={min_val:.6e} max={max_val:.6e}"
                else:
                    minmax_msg = "min=NA max=NA"
                scene_id = video_dict.get("scene_id", None) if isinstance(video_dict, dict) else None
                frame_ids = video_dict.get("frame_ids", None) if isinstance(video_dict, dict) else None
                rank0_print(
                    "[NAN_DEBUG][prepare_inputs_labels_for_multimodal] "
                    f"tag={tag} batch_idx={batch_idx} shape={tuple(t.shape)} dtype={t.dtype} "
                    f"finite_all={finite_all} finite_any={finite_any} nan_count={nan_count} inf_count={inf_count} "
                    f"{minmax_msg} scene_id={scene_id} frame_ids={frame_ids}"
                )

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            _nan_debug_tensor("concat_images_before_encode", concat_images)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)  # [num_frames, num_tokens, embed_dim]
            if mm_branch_fp32:
                encoded_image_features = encoded_image_features.float()
            self._trace_tensor_state("prepare_mm.encoded_image_features", encoded_image_features)
            _nan_debug_tensor("encoded_image_features", encoded_image_features)
            self._retain_and_track_grad("encoded_image_features", encoded_image_features)

            all_faster_video_features = [None] * len(split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat, self.config.mm_spatial_pool_stride))
                    if getattr(self.config, "add_faster_video", False) and self.config.mm_spatial_pool_stride > 1:
                        all_faster_video_features[idx] = self.get_2dPool(image_feat, self.config.mm_spatial_pool_stride // 2)
                else:
                    image_features.append(image_feat)
                _nan_debug_tensor(f"image_features_after_split_idx{idx}", image_features[-1])
                if idx == 0:
                    self._retain_and_track_grad("image_features_after_split_idx0", image_features[-1])
            assert len(image_features) == 1 # only support batch_size=1
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if use_object_proposals:
                object_features = []
                valid_obj_num = 0
                for l in range(obj_num):
                    # print(f"image_featurs: {image_features[0].shape}")
                    # print(f"object_patch: {object_patch[l].shape}")
                    if "patch27" in self.config.object_feature_type:
                        cur_object_features = image_features[0][object_patch[l].view(-1, 196)]
                    elif "patch14" in self.config.object_feature_type:
                        cur_object_features = encoded_image_features[0][object_patch[l].view(-1, 729)]
                    else:
                        raise NotImplementedError

                    if len(cur_object_features) == 0:
                        cur_object_features = torch.zeros(image_features[0].shape[-1]).to(image_features[0].device)
                    else:
                        cur_object_features = cur_object_features.mean(dim=0)
                        valid_obj_num += 1
                    object_features.append(cur_object_features)
                object_features = torch.stack(object_features)
                if use_mlp_pe or use_sin3d_pe:
                    box_center_features = self.get_model().world_position_embedding(object_boxes_center.unsqueeze(0)).squeeze(0)      
                    object_features += box_center_features
            else:
                object_features =  None
            
            if use_sin3d_pe or use_mlp_pe:
                new_image_features = []
                masks = []
                for idx, image_feat in enumerate(image_features):
                    if "discrete" in self.config.world_position_embedding_type:
                        coords = world_coords_discrete[idx].flatten(1, 2)
                    else:
                        coords = world_coords[idx].flatten(1, 2)

                    # replace with mask token
                    if replace_with_mask_token:
                        image_feat, mask = self.replace_with_mask_token(image_feat, getattr(self.config, "view_mask_ratio", 0.))
                    else:
                        image_feat, mask = self.replace_with_mask_token(image_feat, 0.)
                    _nan_debug_tensor(f"image_feat_after_mask_idx{idx}", image_feat)
                    if idx == 0:
                        self._retain_and_track_grad("image_feat_after_mask_idx0", image_feat)

                    # coords: num_frames, num_tokens, 3
                    if os.getenv("ROSS3D_DTYPE_DEBUG", "0") == "1":
                        dtype_dbg_count = int(getattr(self.config, "_dtype_debug_pe_count", 0))
                        dtype_dbg_max = int(os.getenv("ROSS3D_DTYPE_DEBUG_MAX", "2"))
                        if dtype_dbg_count < dtype_dbg_max:
                            pe_feat = self.get_model().world_position_embedding(coords.detach())
                            rank_print(
                                "[DTYPE_DEBUG][prepare_inputs_labels_for_multimodal][world_pe_add] "
                                f"image_feat_before={image_feat.dtype} pe={pe_feat.dtype}"
                            )
                            pe_feat = pe_feat.to(image_feat.dtype)
                            image_feat = image_feat + pe_feat
                            rank_print(
                                "[DTYPE_DEBUG][prepare_inputs_labels_for_multimodal][world_pe_add] "
                                f"image_feat_after={image_feat.dtype}"
                            )
                            setattr(self.config, "_dtype_debug_pe_count", dtype_dbg_count + 1)
                        else:
                            image_feat = image_feat + self.get_model().world_position_embedding(coords.detach()).to(image_feat.dtype)
                    else:
                        image_feat = image_feat + self.get_model().world_position_embedding(coords.detach()).to(image_feat.dtype)
                    _nan_debug_tensor(f"image_feat_after_world_pe_idx{idx}", image_feat)
                    self._trace_tensor_state(f"prepare_mm.image_feat_after_world_pe_idx{idx}", image_feat)
                    if idx == 0:
                        self._retain_and_track_grad("image_feat_after_world_pe_idx0", image_feat)
                    new_image_features.append(image_feat)
                    masks.append(mask)

                image_features = new_image_features

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            (
                                image_feature,
                                boi_ids,
                                eoi_ids,
                                old_image_feature,
                                newline_ids,
                            ) = self.add_token_per_grid(image_feature)
                            _nan_debug_tensor(f"image_feature_after_add_token_per_grid_idx{image_idx}", image_feature)
                            self._trace_tensor_state(f"prepare_mm.image_feature_after_add_token_per_grid_idx{image_idx}", image_feature)
                            if image_idx == 0:
                                self._retain_and_track_grad("image_feature_after_add_token_per_grid_idx0", image_feature)
                            _nan_debug_tensor(f"old_image_feature_after_add_token_per_grid_idx{image_idx}", old_image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type and os.getenv("ROSS3D_DISABLE_IMAGE_NEWLINE_INSERT", "0") != "1":
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            if os.getenv("ROSS3D_DISABLE_IMAGE_NEWLINE_INSERT", "0") != "1":
                                newline_token = self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                self._log_alias_info("prepare_mm.newline_token.anyres", newline_token, self.model.image_newline, "image_newline")
                                newline_token = self._clone_mm_insert_if_debug(newline_token)
                                image_feature = torch.cat((image_feature, newline_token), dim=-1)
                            self._debug_tensor_finite_stats("prepare_mm.image_feature_after_newline_anyres", image_feature)
                            self._check_tensor_finite_or_raise("prepare_mm.image_feature_after_newline_anyres", image_feature)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            if os.getenv("ROSS3D_DISABLE_IMAGE_NEWLINE_INSERT", "0") != "1":
                                newline_token = self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                self._log_alias_info("prepare_mm.newline_token.unpad", newline_token, self.model.image_newline, "image_newline")
                                newline_token = self._clone_mm_insert_if_debug(newline_token)
                                image_feature = torch.cat((image_feature, newline_token), dim=-1)
                            self._debug_tensor_finite_stats("prepare_mm.image_feature_after_newline_unpad", image_feature)
                            self._check_tensor_finite_or_raise("prepare_mm.image_feature_after_newline_unpad", image_feature)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            if os.getenv("ROSS3D_DISABLE_IMAGE_NEWLINE_INSERT", "0") != "1":
                                newline_token = self._clone_mm_insert_if_debug(self.model.image_newline[None])
                                self._log_alias_info("prepare_mm.newline_token.flat", newline_token, self.model.image_newline, "image_newline")
                                image_feature = torch.cat((image_feature, newline_token), dim=0)
                            self._debug_tensor_finite_stats("prepare_mm.image_feature_after_newline_flat", image_feature)
                            self._check_tensor_finite_or_raise("prepare_mm.image_feature_after_newline_flat", image_feature)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
            if mm_branch_fp32:
                image_features = image_features.float()

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_world_coords = []
        cur_image_idx = 0
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            track_first = (batch_idx == 0) and (not bool(getattr(self, "_nan_debug_first_batch_captured", False)))
            self._nan_debug_track_this_batch = track_first
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            assert num_images == 1

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)

            # Hanwen
            # Add input coord PE
            if hasattr(self.config, "coord_token_ids") and (use_sin3d_pe or use_mlp_pe):
                query_coord_tokens = (cat_cur_input_ids_noim == self.config.coord_token_ids[0])

                # Only apply world position embedding if everything we need is present
                if (
                    query_coord_tokens.sum() != 0
                    and box_input is not None
                    and hasattr(self.get_model(), "world_position_embedding")
                ):
                    box_tensor = box_input.unsqueeze(0).detach()
                    box_tensor = box_tensor.to(
                        device=cur_input_embeds.device,
                        dtype=cur_input_embeds.dtype,
                    )

                    cur_input_embeds = cur_input_embeds.clone()
                    cur_input_embeds[query_coord_tokens] = (
                        cur_input_embeds[query_coord_tokens]
                        + self.get_model().world_position_embedding(box_tensor)[:, 0]
                    )
                # else: no valid world-position info for this sample → skip instead of crashing


            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_world_coords = []
            cur_pos_index = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                text_len = cur_input_embeds_no_im[i].shape[0]
                cur_new_labels.append(cur_labels_noim[i])
                if use_mrope_position_embedding:
                    cur_new_world_coords.append(
                        torch.arange(cur_pos_index, cur_pos_index + len(cur_input_embeds_no_im[i])).to(cur_input_embeds_no_im[i].device).unsqueeze(1).repeat(1, 3)
                    )
                    cur_pos_index += len(cur_input_embeds_no_im[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                        boi_ids = list(map(lambda x: x + text_len, boi_ids))
                        eoi_ids = list(map(lambda x: x + text_len, eoi_ids))
                        newline_ids = list(map(lambda x: x + text_len, newline_ids))
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]

                    if getattr(self.config, "verbose_logging", False):
                        rank0_print(
                            "[token_count] "
                            f"text={text_len}, "
                            f"image={cur_image_features.shape[0]}, "
                            f"total={text_len + cur_image_features.shape[0]}, "
                            f"model_max_length={tokenizer_model_max_length}"
                        )
                    
                    if use_mrope_position_embedding:
                        coords = world_coords_discrete[batch_idx]
                        V, H, W, D = coords.shape
                        new_coords = torch.zeros(V*H*(W+1), 3).to(cur_input_embeds_no_im[i].device).view(V, H, W+1, 3)
                        new_coords[:, :, :W, :] = coords
                        new_coords = new_coords.view(-1, 3)
                        cur_pos_index += V * H * (W + 1)
                        cur_new_world_coords.append(new_coords)

                    cur_image_idx += 1
                    if mm_branch_fp32:
                        cur_image_features = cur_image_features.to(cur_input_embeds.dtype)
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            target_device = cur_input_embeds.device
            cur_new_input_embeds = [x.to(target_device) for x in cur_new_input_embeds]
            for seq_idx, cur_seq in enumerate(cur_new_input_embeds):
                _nan_debug_tensor(f"cur_new_input_embeds_list_before_cat_idx{seq_idx}", cur_seq, batch_idx=batch_idx)
                self._trace_tensor_state(f"prepare_mm.cur_new_input_embeds_list_before_cat_idx{seq_idx}", cur_seq)

            # import pdb; pdb.set_trace()
            if os.getenv("ROSS3D_CLONE_MM_INSERTS_DEBUG", "0") == "1":
                cur_new_input_embeds = [x.clone() for x in cur_new_input_embeds]
            for seq_idx, cur_seq in enumerate(cur_new_input_embeds):
                self._check_tensor_finite_or_raise(f"prepare_mm.cur_new_input_embeds_before_cat_idx{seq_idx}", cur_seq)
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            _nan_debug_tensor("cur_new_input_embeds_after_cat", cur_new_input_embeds, batch_idx=batch_idx)
            self._debug_tensor_finite_stats("prepare_mm.cur_new_input_embeds_after_cat", cur_new_input_embeds)
            self._trace_tensor_state("prepare_mm.cur_new_input_embeds_after_cat", cur_new_input_embeds)
            self._check_tensor_finite_or_raise("prepare_mm.cur_new_input_embeds_after_cat", cur_new_input_embeds)
            self._retain_and_track_grad(f"packed_embeds_batch{batch_idx}", cur_new_input_embeds)
            if batch_idx == 0:
                self._retain_and_track_grad("cur_new_input_embeds_after_cat", cur_new_input_embeds)

            boi_ids_tensor = torch.LongTensor(boi_ids)
            eoi_ids_tensor = torch.LongTensor(eoi_ids)
            newline_ids_tensor = torch.LongTensor(newline_ids)
            if nan_dbg_enabled:
                records = getattr(self, "_nan_debug_packed_grad_records", None)
                if records is None:
                    records = []
                    setattr(self, "_nan_debug_packed_grad_records", records)
                records.append({
                    "packed": cur_new_input_embeds,
                    "newline_ids": list(newline_ids),
                    "boi_ids": list(boi_ids),
                    "eoi_ids": list(eoi_ids),
                })

            boi_slice_check = cur_new_input_embeds[boi_ids_tensor]
            eoi_slice_check = cur_new_input_embeds[eoi_ids_tensor]
            newline_slice_check = cur_new_input_embeds[newline_ids_tensor]
            fp32_layout_tolerant = os.getenv("ROSS3D_MM_BRANCH_FP32_DEBUG", "0") == "1"
            if fp32_layout_tolerant:
                atol = float(os.getenv("ROSS3D_MM_LAYOUT_ATOL", "1e-2"))
                rtol = float(os.getenv("ROSS3D_MM_LAYOUT_RTOL", "1e-2"))
                first_match = torch.isclose(
                    old_image_feature[:, 0].float(),
                    boi_slice_check.float(),
                    atol=atol,
                    rtol=rtol,
                ).all(dim=1)
                last_match = torch.isclose(
                    old_image_feature[:, -1].float(),
                    eoi_slice_check.float(),
                    atol=atol,
                    rtol=rtol,
                ).all(dim=1)
                newline_match = torch.isclose(
                    self.model.image_newline.unsqueeze(0).repeat(len(newline_ids), 1).float(),
                    newline_slice_check.float(),
                    atol=atol,
                    rtol=rtol,
                ).all(dim=1)
            else:
                first_match = torch.all(old_image_feature[:, 0] == boi_slice_check, dim=1)
                last_match = torch.all(old_image_feature[:, -1] == eoi_slice_check, dim=1)
                newline_match = torch.all(
                    self.model.image_newline.unsqueeze(0).repeat(len(newline_ids), 1) == newline_slice_check,
                    dim=1,
                )

            ok_first = bool(torch.all(first_match).item())
            ok_last = bool(torch.all(last_match).item())
            ok_newline = bool(torch.all(newline_match).item())

            _nan_debug_tensor("old_image_feature_before_assert", old_image_feature, batch_idx=batch_idx)
            _nan_debug_tensor("cur_new_input_embeds_before_assert", cur_new_input_embeds, batch_idx=batch_idx)
            boi_slice = boi_slice_check
            eoi_slice = eoi_slice_check
            newline_slice = newline_slice_check if len(newline_ids) > 0 else None
            _nan_debug_tensor("cur_new_input_embeds_boi_before_assert", boi_slice, batch_idx=batch_idx)
            self._trace_tensor_state("prepare_mm.boi_slice", boi_slice)
            _nan_debug_tensor("cur_new_input_embeds_eoi_before_assert", eoi_slice, batch_idx=batch_idx)
            self._trace_tensor_state("prepare_mm.eoi_slice", eoi_slice)
            _nan_debug_tensor("cur_new_input_embeds_newline_before_assert", newline_slice, batch_idx=batch_idx)
            self._trace_tensor_state("prepare_mm.newline_slice", newline_slice)
            if batch_idx == 0:
                self._retain_and_track_grad("boi_slice", boi_slice)
                self._retain_and_track_grad("eoi_slice", eoi_slice)
                self._retain_and_track_grad("newline_slice", newline_slice)
                if nan_dbg_enabled:
                    self._nan_debug_layout_snapshot = {
                        "old_image_feature": old_image_feature,
                        "cur_new_input_embeds": cur_new_input_embeds,
                        "boi_ids": boi_ids_tensor,
                        "eoi_ids": eoi_ids_tensor,
                        "newline_ids": newline_ids_tensor if len(newline_ids) > 0 else None,
                        "image_newline": getattr(self.model, "image_newline", None),
                    }
                self._nan_debug_first_batch_captured = True
            _nan_debug_tensor("model_image_newline_before_assert", self.model.image_newline, batch_idx=batch_idx)

            if (not ok_first or not ok_last or not ok_newline) and os.getenv("ROSS3D_PACK_DEBUG", "0") == "1":
                is_rank0 = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    is_rank0 = torch.distributed.get_rank() == 0

                if is_rank0:
                    k = 8
                    first_fail = torch.where(~first_match)[0].tolist()
                    last_fail = torch.where(~last_match)[0].tolist()
                    newline_fail = torch.where(~newline_match)[0].tolist()

                    patch_h = int(math.ceil(math.sqrt(self.model.image_embed_len)))
                    patch_w = int(math.ceil(float(self.model.image_embed_len) / max(patch_h, 1)))
                    frame_count = int(old_image_feature.shape[0]) if old_image_feature is not None else -1
                    token_count = int(old_image_feature.shape[1]) if old_image_feature is not None else -1
                    rank0_print(
                        "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][assert_fail] "
                        f"batch_idx={batch_idx} ok_first={ok_first} ok_last={ok_last} ok_newline={ok_newline} "
                        f"cur_new_input_embeds_shape={tuple(cur_new_input_embeds.shape)} old_image_feature_shape={tuple(old_image_feature.shape)} "
                        f"len_boi={len(boi_ids)} len_eoi={len(eoi_ids)} len_newline={len(newline_ids)} "
                        f"text_len={text_len} num_images={int(num_images)} cur_image_idx={int(cur_image_idx)} "
                        f"mm_patch_merge_type={mm_patch_merge_type} mm_newline_position={mm_newline_position} "
                        f"patch_h={patch_h} patch_w={patch_w} frame_count={frame_count} token_count={token_count} "
                        f"use_object_proposals={use_object_proposals} "
                        f"scene_id={video_dict.get('scene_id', None) if isinstance(video_dict, dict) else None} "
                        f"frame_ids={video_dict.get('frame_ids', None) if isinstance(video_dict, dict) else None} "
                        f"video_dict_keys={sorted(list(video_dict.keys())) if isinstance(video_dict, dict) else None}"
                    )
                    rank0_print(
                        "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][assert_fail][indices] "
                        f"first_fail={first_fail} last_fail={last_fail} newline_fail={newline_fail}"
                    )
                    rank0_print(
                        "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][assert_fail][id_head] "
                        f"boi_ids={boi_ids[:k]} eoi_ids={eoi_ids[:k]} newline_ids={newline_ids[:k]}"
                    )

                    for logical_idx in first_fail[:k]:
                        packed_idx = int(boi_ids_tensor[logical_idx].item())
                        expected = old_image_feature[logical_idx, 0]
                        actual = cur_new_input_embeds[packed_idx]
                        diff = (expected - actual).abs()
                        rank0_print(
                            "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][first_mismatch] "
                            f"logical_idx={int(logical_idx)} packed_idx={packed_idx} "
                            f"max_abs_diff={float(diff.max().item()):.6e} mean_abs_diff={float(diff.mean().item()):.6e} "
                            f"expected_dtype={expected.dtype} actual_dtype={actual.dtype}"
                        )

                    if len(first_fail) > 0:
                        fi = int(first_fail[0])
                        f_expected = old_image_feature[fi, 0]
                        f_actual = cur_new_input_embeds[int(boi_ids_tensor[fi].item())]
                        rank0_print(
                            "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][first_nan_profile] "
                            f"logical_idx={fi} expected_has_nan={bool(torch.isnan(f_expected).any().item())} "
                            f"actual_has_nan={bool(torch.isnan(f_actual).any().item())} "
                            f"expected_finite_ratio={float(torch.isfinite(f_expected).float().mean().item()):.6f} "
                            f"actual_finite_ratio={float(torch.isfinite(f_actual).float().mean().item()):.6f}"
                        )

                    for logical_idx in last_fail[:k]:
                        packed_idx = int(eoi_ids_tensor[logical_idx].item())
                        expected = old_image_feature[logical_idx, -1]
                        actual = cur_new_input_embeds[packed_idx]
                        diff = (expected - actual).abs()
                        rank0_print(
                            "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][last_mismatch] "
                            f"logical_idx={int(logical_idx)} packed_idx={packed_idx} "
                            f"max_abs_diff={float(diff.max().item()):.6e} mean_abs_diff={float(diff.mean().item()):.6e} "
                            f"expected_dtype={expected.dtype} actual_dtype={actual.dtype}"
                        )

                    if len(last_fail) > 0:
                        li = int(last_fail[0])
                        l_expected = old_image_feature[li, -1]
                        l_actual = cur_new_input_embeds[int(eoi_ids_tensor[li].item())]
                        rank0_print(
                            "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][last_nan_profile] "
                            f"logical_idx={li} expected_has_nan={bool(torch.isnan(l_expected).any().item())} "
                            f"actual_has_nan={bool(torch.isnan(l_actual).any().item())} "
                            f"expected_finite_ratio={float(torch.isfinite(l_expected).float().mean().item()):.6f} "
                            f"actual_finite_ratio={float(torch.isfinite(l_actual).float().mean().item()):.6f}"
                        )

                    newline_expected = self.model.image_newline
                    for logical_idx in newline_fail[:k]:
                        packed_idx = int(newline_ids_tensor[logical_idx].item())
                        actual = cur_new_input_embeds[packed_idx]
                        diff = (newline_expected - actual).abs()
                        rank0_print(
                            "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][newline_mismatch] "
                            f"logical_idx={int(logical_idx)} packed_idx={packed_idx} "
                            f"max_abs_diff={float(diff.max().item()):.6e} mean_abs_diff={float(diff.mean().item()):.6e} "
                            f"expected_dtype={newline_expected.dtype} actual_dtype={actual.dtype}"
                        )

                    if len(newline_fail) > 0:
                        ni = int(newline_fail[0])
                        n_actual = cur_new_input_embeds[int(newline_ids_tensor[ni].item())]
                        rank0_print(
                            "[PACK_DEBUG][prepare_inputs_labels_for_multimodal][newline_nan_profile] "
                            f"logical_idx={ni} expected_has_nan={bool(torch.isnan(newline_expected).any().item())} "
                            f"actual_has_nan={bool(torch.isnan(n_actual).any().item())} "
                            f"expected_finite_ratio={float(torch.isfinite(newline_expected).float().mean().item()):.6f} "
                            f"actual_finite_ratio={float(torch.isfinite(n_actual).float().mean().item()):.6f}"
                        )

            if not (ok_first and ok_last and ok_newline):
                raise RuntimeError(
                    f"Bad multimodal layout: ok_first={ok_first}, "
                    f"ok_last={ok_last}, ok_newline={ok_newline}"
                )

            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            if use_mrope_position_embedding:
                cur_new_world_coords = torch.cat(cur_new_world_coords, dim=0)
                new_world_coords.append(cur_new_world_coords)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mrope_position_ids = torch.zeros((batch_size, max_len, 3), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, -cur_len:, :] = new_world_coords[i][-cur_len:, :]

            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, :cur_len, :] = new_world_coords[i][:cur_len, :]

        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        if use_mrope_position_embedding:
            position_ids = mrope_position_ids

        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        self._debug_tensor_finite_stats("prepare_mm.mask_token.end", getattr(self.get_model(), "mask_token", None))
        self._debug_tensor_finite_stats("prepare_mm.image_newline.end", getattr(self.get_model(), "image_newline", None))
        self._debug_tensor_finite_stats("prepare_mm.new_input_embeds.end", new_input_embeds)
        self._check_tensor_finite_or_raise("prepare_mm.mask_token.end", getattr(self.get_model(), "mask_token", None))
        self._check_tensor_finite_or_raise("prepare_mm.image_newline.end", getattr(self.get_model(), "image_newline", None))
        self._check_tensor_finite_or_raise("prepare_mm.new_input_embeds.end", new_input_embeds)
        try:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes, boi_ids, eoi_ids, newline_ids, masks[0]
        except:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes, boi_ids, eoi_ids, newline_ids, None

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def _extract_frame_patch_hidden_states(
        self,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-frame patch tokens using the same ordering as VM/BEV loss."""
        B = hidden_states.shape[0]
        assert B == 1, "Per-frame patch extraction currently supports batch_size==1"

        boi_ids_tensor = torch.LongTensor(boi_ids)
        eoi_ids_tensor = torch.LongTensor(eoi_ids)
        T = boi_ids_tensor.shape[0]
        P = self.model.image_embed_len
        D = hidden_states.shape[-1]
        patch_h = math.ceil(math.sqrt(P))

        feats = torch.zeros((T, P, D), dtype=hidden_states.dtype, device=hidden_states.device)

        for t, (cur_boi, cur_eoi) in enumerate(zip(boi_ids_tensor, eoi_ids_tensor)):
            if (cur_boi is None) or (cur_eoi is None):
                continue

            cur_rows = [hidden_states[0][cur_boi: newline_ids[t * patch_h]]]
            for k in range(t * patch_h + 1, (t + 1) * patch_h):
                cur_rows.append(hidden_states[0][newline_ids[k - 1] + 1: newline_ids[k]])
            cur_rows.append(hidden_states[0][newline_ids[(t + 1) * patch_h - 1] + 1: cur_eoi])

            feats[t] = torch.cat(cur_rows, dim=0)

        return feats

    @torch._dynamo.disable
    def extract_occupancy_object_embeddings(
        self,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        video_dict: Optional[Dict[str, torch.Tensor]] = None,
        global_step: Optional[int] = "NA",
        eps: float = 1e-6,
        need_patch_embeddings: bool = True,
        need_geom_metadata: bool = True,
        need_temp_metadata: bool = True,
    ) -> Optional[Dict[str, Union[torch.Tensor, List[int], List[str], str]]]:
        first_extract_fail = None
        used_patch_proj = False
        used_obj_norm = False

        def _extract_return(reason: str, value):
            nonlocal first_extract_fail
            if first_extract_fail is None:
                first_extract_fail = reason
            setattr(self, "_occ_dbg_extract_return_reason", reason)
            setattr(self, "_occ_dbg_used_patch_proj", bool(used_patch_proj))
            setattr(self, "_occ_dbg_used_obj_norm", bool(used_obj_norm))
            rlog(f"OCC_DECISION step={global_step} fn=extract guard={reason} pass=0")
            rlog(f"FIRST_FAIL step={global_step} fn=extract first_fail={first_extract_fail or 'none'}")
            return value

        if not hasattr(self.model, "occupancy_patch_projector") or not hasattr(self.model, "occupancy_object_norm"):
            return _extract_return("missing_projector", None)
        if video_dict is None:
            return _extract_return("video_dict_none", None)
        patch_occupancy = video_dict.get("patch_occupancy", None)
        if patch_occupancy is None:
            return _extract_return("patch_occupancy_none", None)
        rlog(f"OCC_DECISION step={global_step} fn=extract guard=global_guards pass=1")

        patch_feats = self._extract_frame_patch_hidden_states(hidden_states, boi_ids, eoi_ids, newline_ids)
        use_occ_patch_proj = bool(getattr(self.config, "use_occupancy_patch_projector", True))
        if use_occ_patch_proj:
            rlog(f"PARAM_USE step={global_step} module=occupancy_patch_projector used=1")
            used_patch_proj = True
            projected_patch_feats = self.model.occupancy_patch_projector(patch_feats)
        else:
            rlog(f"PARAM_USE step={global_step} module=occupancy_patch_projector used=0 reason=ablation_disabled")
            target_dim = int(self.model.occupancy_object_norm.normalized_shape[0])
            projected_patch_feats = self._occ_match_last_dim(patch_feats, target_dim)
        rlog(f"PATCH_PROJECTOR_OUTPUT shape={tshape(projected_patch_feats)}")
        rlog("EXTRACT_AFTER_PATCH_PROJECTOR")
        self._log_occ_cuda_memory("aux_after_patch_projector")

        frame_ids = video_dict.get("frame_ids", [str(i) for i in range(projected_patch_feats.shape[0])])
        scene_id = video_dict.get("scene_id", None)
        visible_bboxes = video_dict.get("visible_bboxes", [None for _ in range(projected_patch_feats.shape[0])])

        T, P, Dp = projected_patch_feats.shape
        device = projected_patch_feats.device
        dtype = projected_patch_feats.dtype

        detected_ids_per_frame = []
        all_detected_ids = set()
        obj_id_to_label = {}
        rlog("EXTRACT_BEFORE_FRAME_LOOP")
        for fidx in range(T):
            occ_anno = patch_occupancy[fidx] if fidx < len(patch_occupancy) else None
            vis_anno = visible_bboxes[fidx] if fidx < len(visible_bboxes) else None
            n_visible = len(vis_anno.get("detected", [])) if vis_anno is not None else 0

            # collect labels from visible_bboxes first
            if vis_anno is not None:
                for obj in vis_anno.get("detected", []):
                    try:
                        oid = int(obj.get("object_id"))
                    except Exception:
                        continue
                    label = str(obj.get("label", "")).strip().lower()
                    if label:
                        if oid not in obj_id_to_label:
                            obj_id_to_label[oid] = label
                        elif obj_id_to_label[oid] != label:
                            rank0_print(
                                f"[occupancy_aux] conflicting labels for object_id={oid}: "
                                f"keep='{obj_id_to_label[oid]}', ignore='{label}'"
                            )

            if occ_anno is None:
                detected_ids_per_frame.append([])
                rlog(f"EXTRACT_FRAME_SUMMARY frame={fidx} visible_boxes={n_visible} valid_ids=0 kept=0")
                continue

            detected_objects = occ_anno.get("detected_objects", [])
            detected_ids = []
            for obj in detected_objects:
                if "object_id" in obj:
                    try:
                        oid = int(obj["object_id"])
                        detected_ids.append(oid)
                        if oid not in obj_id_to_label:
                            label = str(obj.get("label", "")).strip().lower()
                            if label:
                                obj_id_to_label[oid] = label
                    except Exception:
                        continue
            detected_ids_per_frame.append(detected_ids)
            all_detected_ids.update(detected_ids)
        rlog("EXTRACT_AFTER_FRAME_LOOP")
        obj_ids_union = sorted(all_detected_ids)
        O = len(obj_ids_union)
        rlog(f"OCC_DECISION step={global_step} fn=extract guard=has_union_objects pass={1 if O > 0 else 0}")

        empty_embeddings = torch.zeros((T, O, Dp), device=device, dtype=dtype)
        empty_present = torch.zeros((T, O), device=device, dtype=torch.bool)

        if O == 0:
            outputs = {
                "object_embeddings": empty_embeddings,
                "present": empty_present,
            }
            if need_patch_embeddings:
                outputs["patch_embeddings"] = projected_patch_feats
            if need_geom_metadata:
                outputs["obj_ids_union"] = torch.zeros((0,), device=device, dtype=torch.long)
                outputs["obj_labels_union"] = []
                outputs["frame_ids"] = frame_ids
                outputs["scene_id"] = scene_id
            if need_temp_metadata:
                outputs["obj_cat_ids_union"] = torch.zeros((0,), device=device, dtype=torch.long)
            return _extract_return("no_union_objects", outputs)

        id_to_union_col = {oid: u for u, oid in enumerate(obj_ids_union)}

        valid_frame_count = 0
        _object_norm_logged = False
        z_frames = []
        present_frames = []
        for fidx in range(T):
            occ_anno = patch_occupancy[fidx] if fidx < len(patch_occupancy) else None
            occ_matrix_union = torch.zeros((P, O), device=device, dtype=dtype)
            if occ_anno is not None:
                patches = occ_anno.get("patches", [])
                for patch in patches:
                    pidx = int(patch.get("patch_index", -1))
                    if pidx < 0 or pidx >= P:
                        continue
                    patch_occ = patch.get("occupancy", {})
                    for k, v in patch_occ.items():
                        try:
                            oid = int(k)
                        except Exception:
                            continue
                        if oid not in id_to_union_col:
                            continue
                        occ_matrix_union[pidx, id_to_union_col[oid]] = float(v)

            occ_sum_union = occ_matrix_union.sum(dim=0)
            valid_union = occ_sum_union > 0
            z_local = (occ_matrix_union.T @ projected_patch_feats[fidx]) / (occ_sum_union[:, None] + eps)
            if not _object_norm_logged:
                rlog(f"PARAM_USE step={global_step} module=occupancy_object_norm used=1")
                used_obj_norm = True
            z_local = self.model.occupancy_object_norm(z_local)
            if not _object_norm_logged:
                _object_norm_logged = True

            z_frame = torch.where(valid_union[:, None], z_local, torch.zeros_like(z_local))
            present_frame = valid_union.to(dtype=torch.bool)

            if valid_union.any():
                valid_frame_count += 1
            z_frames.append(z_frame)
            present_frames.append(present_frame)

        object_embeddings = torch.stack(z_frames, dim=0)
        present = torch.stack(present_frames, dim=0)

        if valid_frame_count == 0:
            if first_extract_fail is None:
                first_extract_fail = "no_valid_frames"
            rlog(f"OCC_DECISION step={global_step} fn=extract guard=no_valid_frames pass=0")

        self._log_occ_cuda_memory(f"aux_after_object_embeddings T={T} P={P} O={O} Dp={Dp}")

        obj_labels_union = []
        for oid in obj_ids_union:
            label = obj_id_to_label.get(oid, f"__unknown_obj_{oid}")
            obj_labels_union.append(label)
        label_to_cat_id = {}
        obj_cat_ids = []
        for label in obj_labels_union:
            if label not in label_to_cat_id:
                label_to_cat_id[label] = len(label_to_cat_id)
            obj_cat_ids.append(label_to_cat_id[label])

        outputs = {
            "object_embeddings": object_embeddings,
            "present": present,
        }
        if need_patch_embeddings:
            outputs["patch_embeddings"] = projected_patch_feats
        if need_geom_metadata:
            outputs["obj_ids_union"] = torch.tensor(obj_ids_union, device=device, dtype=torch.long)
            outputs["obj_labels_union"] = obj_labels_union
            outputs["frame_ids"] = frame_ids
            outputs["scene_id"] = scene_id
        if need_temp_metadata:
            outputs["obj_cat_ids_union"] = torch.tensor(obj_cat_ids, device=device, dtype=torch.long)
        setattr(self, "_occ_dbg_extract_return_reason", None)
        setattr(self, "_occ_dbg_used_patch_proj", bool(used_patch_proj))
        setattr(self, "_occ_dbg_used_obj_norm", bool(used_obj_norm))
        rlog(f"FIRST_FAIL step={global_step} fn=extract first_fail={first_extract_fail or 'none'}")
        return outputs

    @torch._dynamo.disable
    def _log_occ_cuda_memory(self, tag: str) -> None:
        if not getattr(self.config, "verbose_logging", False):
            return
        if not getattr(self.config, "occ_debug_memory", False):
            return
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        rank0_print(
            "[occupancy][cuda_mem] "
            f"{tag}: allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, "
            f"max_allocated={max_alloc:.2f}GB"
        )

    def _build_patch_centers_normalized(
        self,
        occ_anno: Dict[str, Any],
        device,
        dtype,
        target_grid: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        preprocess = occ_anno.get("vision_tower_preprocess", {})
        grid = preprocess.get("grid", None)
        if grid is None:
            raise ValueError("Missing vision_tower_preprocess.grid in occupancy annotation.")
        src_h, src_w = int(grid[0]), int(grid[1])
        if target_grid is None:
            H, W = src_h, src_w
        else:
            H, W = int(target_grid[0]), int(target_grid[1])
        patches = occ_anno.get("patches", [])
        P = H * W
        patch_centers = torch.zeros((P, 2), device=device, dtype=dtype)
        if target_grid is None:
            for patch in patches:
                pidx = int(patch.get("patch_index", -1))
                if pidx < 0 or pidx >= P:
                    continue
                row = int(patch.get("row", pidx // W))
                col = int(patch.get("col", pidx % W))
                patch_centers[pidx, 0] = (col + 0.5) / float(W)
                patch_centers[pidx, 1] = (row + 0.5) / float(H)
            if len(patches) == 0:
                rows = torch.arange(H, device=device, dtype=dtype)
                cols = torch.arange(W, device=device, dtype=dtype)
                rr, cc = torch.meshgrid(rows, cols, indexing="ij")
                patch_centers[:, 0] = (cc.reshape(-1) + 0.5) / float(W)
                patch_centers[:, 1] = (rr.reshape(-1) + 0.5) / float(H)
            return patch_centers

        rows = torch.arange(H, device=device, dtype=dtype)
        cols = torch.arange(W, device=device, dtype=dtype)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        patch_centers[:, 0] = (cc.reshape(-1) + 0.5) / float(W)
        patch_centers[:, 1] = (rr.reshape(-1) + 0.5) / float(H)
        return patch_centers

    def _convert_visible_bbox_to_geom_target(self, vis_obj: Dict[str, Any], vis_anno: Dict[str, Any], occ_anno: Dict[str, Any]):
        image_size = vis_anno.get("image_size", None)
        if image_size is None or len(image_size) < 2:
            return None, None, None
        raw_w, raw_h = float(image_size[0]), float(image_size[1])
        if raw_w <= 0 or raw_h <= 0:
            return None, None, None

        preprocess = occ_anno.get("vision_tower_preprocess", {})
        if not preprocess and isinstance(vis_anno, dict):
            preprocess = vis_anno.get("vision_tower_preprocess", {})
        resize = preprocess.get("resize", None)
        grid = preprocess.get("grid", None)
        if resize is None or grid is None:
            raise ValueError("Missing resize/grid in occupancy annotation for geometry conversion.")
        img_h, img_w = float(resize[0]), float(resize[1])

        bbox = vis_obj.get("bbox_xyxy", None)
        if bbox is None or len(bbox) < 4:
            return None, None, None
        x1_raw, y1_raw, x2_raw, y2_raw = [float(v) for v in bbox[:4]]
        x1_img = x1_raw * (img_w / raw_w)
        y1_img = y1_raw * (img_h / raw_h)
        x2_img = x2_raw * (img_w / raw_w)
        y2_img = y2_raw * (img_h / raw_h)

        x1n = x1_img / img_w
        y1n = y1_img / img_h
        x2n = x2_img / img_w
        y2n = y2_img / img_h

        x1n = max(0.0, min(1.0, x1n))
        y1n = max(0.0, min(1.0, y1n))
        x2n = max(0.0, min(1.0, x2n))
        y2n = max(0.0, min(1.0, y2n))

        if x2n <= x1n or y2n <= y1n:
            return None, None, None

        center_norm = None
        center_uv = vis_obj.get("projected_center_uv", None)
        if center_uv is not None and len(center_uv) >= 2:
            cx_raw, cy_raw = float(center_uv[0]), float(center_uv[1])
            cx_img = cx_raw * (img_w / raw_w)
            cy_img = cy_raw * (img_h / raw_h)
            center_norm = [
                max(0.0, min(1.0, cx_img / img_w)),
                max(0.0, min(1.0, cy_img / img_h)),
            ]

        center_visible = bool(vis_obj.get("projected_center_in_bbox", False))
        return [x1n, y1n, x2n, y2n], center_norm, center_visible

    def _build_frame_object_occupancy_targets(
        self,
        occ_anno: Dict[str, Any],
        obj_ids_union: torch.Tensor,
        device,
        dtype,
        target_grid: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        preprocess = occ_anno.get("vision_tower_preprocess", {})
        grid = preprocess.get("grid", None)
        if grid is None:
            raise ValueError("Missing vision_tower_preprocess.grid in occupancy annotation.")
        src_h, src_w = int(grid[0]), int(grid[1])
        if target_grid is None:
            H, W = src_h, src_w
        else:
            H, W = int(target_grid[0]), int(target_grid[1])
        P = H * W
        O = int(obj_ids_union.shape[0])
        occ_target = torch.zeros((P, O), device=device, dtype=dtype)

        detected_objects = occ_anno.get("detected_objects", [])
        detected_ids = []
        for obj in detected_objects:
            if "object_id" in obj:
                try:
                    detected_ids.append(int(obj["object_id"]))
                except Exception:
                    continue
        if len(detected_ids) == 0 or O == 0:
            return occ_target

        id_to_union_col = {int(oid): i for i, oid in enumerate(obj_ids_union.detach().cpu().tolist())}
        local_to_union = {}
        for j, oid in enumerate(detected_ids):
            if oid in id_to_union_col:
                local_to_union[j] = id_to_union_col[oid]
        if len(local_to_union) == 0:
            return occ_target

        rows, cols, vals = [], [], []
        for patch in occ_anno.get("patches", []):
            src_pidx = int(patch.get("patch_index", -1))
            src_row = int(patch.get("row", src_pidx // max(src_w, 1)))
            src_col = int(patch.get("col", src_pidx % max(src_w, 1)))
            if target_grid is None:
                pidx = src_pidx
                if pidx < 0 or pidx >= P:
                    continue
            else:
                if src_row < 0 or src_row >= src_h or src_col < 0 or src_col >= src_w:
                    continue
                cx = (src_col + 0.5) / float(src_w)
                cy = (src_row + 0.5) / float(src_h)
                tgt_col = min(max(int(cx * W), 0), W - 1)
                tgt_row = min(max(int(cy * H), 0), H - 1)
                pidx = tgt_row * W + tgt_col
            patch_occ = patch.get("occupancy", {})
            for k, v in patch_occ.items():
                try:
                    oid = int(k)
                except Exception:
                    continue
                if oid not in id_to_union_col:
                    continue
                rows.append(pidx)
                cols.append(id_to_union_col[oid])
                vals.append(float(v))

        if len(rows) > 0:
            row_t = torch.tensor(rows, device=device, dtype=torch.long)
            col_t = torch.tensor(cols, device=device, dtype=torch.long)
            val_t = torch.tensor(vals, device=device, dtype=dtype)
            occ_target.index_put_((row_t, col_t), val_t, accumulate=False)

        return occ_target

    def _generalized_iou_xyxy(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

        inter_w = (x2 - x1).clamp(min=0.0)
        inter_h = (y2 - y1).clamp(min=0.0)
        inter = inter_w * inter_h

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
        union = area1 + area2 - inter
        iou = inter / (union + eps)

        cx1 = torch.minimum(boxes1[:, 0], boxes2[:, 0])
        cy1 = torch.minimum(boxes1[:, 1], boxes2[:, 1])
        cx2 = torch.maximum(boxes1[:, 2], boxes2[:, 2])
        cy2 = torch.maximum(boxes1[:, 3], boxes2[:, 3])
        c_area = (cx2 - cx1).clamp(min=0.0) * (cy2 - cy1).clamp(min=0.0)
        giou = iou - (c_area - union) / (c_area + eps)
        return giou

    @torch._dynamo.disable
    def compute_occupancy_geometry_loss(
        self,
        occupancy_aux_outputs: Dict[str, Any],
        video_dict: Dict[str, Any],
        global_step: Optional[int] = "NA",
    ) -> torch.Tensor:
        for _attr in [
            "_occ_geom_mask_loss",
            "_occ_geom_box_loss",
            "_occ_geom_ctr_loss",
            "_occ_geom_vis_loss",
            "_occ_geom_mask_bce_loss",
            "_occ_geom_mask_dice_loss",
            "_occ_geom_box_l1_loss",
            "_occ_geom_box_giou_loss",
        ]:
            setattr(self, _attr, None)
        occ_geom_debug = os.getenv("ROSS3D_OCC_GEOM_DEBUG", "0") == "1"
        occ_geom_debug_max_steps = int(os.getenv("ROSS3D_OCC_GEOM_DEBUG_MAX_STEPS", "200"))
        should_log_occ_geom = occ_geom_debug
        if should_log_occ_geom:
            try:
                should_log_occ_geom = int(global_step) < occ_geom_debug_max_steps
            except Exception:
                should_log_occ_geom = True

        scene_id = "NA"
        if isinstance(video_dict, dict):
            scene_id = video_dict.get("scene_id", "NA")

        dbg_stats = {
            "frames_total": 0,
            "frames_with_both_annos": 0,
            "frames_missing_preprocess": 0,
            "frames_grid_mismatch_remapped": 0,
            "frames_invalid_target_grid": 0,
            "frames_valid_geom_meta": 0,
            "frames_with_candidates": 0,
            "vis_total": 0,
            "vis_missing_object_id": 0,
            "vis_in_union": 0,
            "vis_present": 0,
            "vis_bbox_valid": 0,
            "vis_bbox_invalid": 0,
            "selected_pairs": 0,
        }

        def _emit_occ_geom_debug(status: str, reason: Optional[str] = None):
            if not should_log_occ_geom:
                return
            msg = (
                f"[OCC_GEOM_DEBUG] step={global_step} scene_id={scene_id} status={status} "
                f"reason={reason if reason is not None else 'none'} "
                f"frames_total={dbg_stats['frames_total']} "
                f"frames_with_both_annos={dbg_stats['frames_with_both_annos']} "
                f"frames_missing_preprocess={dbg_stats['frames_missing_preprocess']} "
                f"frames_grid_mismatch_remapped={dbg_stats['frames_grid_mismatch_remapped']} "
                f"frames_invalid_target_grid={dbg_stats['frames_invalid_target_grid']} "
                f"frames_valid_geom_meta={dbg_stats['frames_valid_geom_meta']} "
                f"frames_with_candidates={dbg_stats['frames_with_candidates']} "
                f"vis_total={dbg_stats['vis_total']} "
                f"vis_missing_object_id={dbg_stats['vis_missing_object_id']} "
                f"vis_in_union={dbg_stats['vis_in_union']} "
                f"vis_present={dbg_stats['vis_present']} "
                f"vis_bbox_valid={dbg_stats['vis_bbox_valid']} "
                f"vis_bbox_invalid={dbg_stats['vis_bbox_invalid']} "
                f"selected_pairs={dbg_stats['selected_pairs']}"
            )
            rank_print(msg)

        first_geom_fail = None
        used_geom_any = False
        if not all(
            hasattr(self.model, name)
            for name in [
                "occ_geom_patch_norm",
                "occ_geom_obj_query",
                "occ_geom_relation",
                "occ_geom_mask_head",
                "occ_geom_center_head",
                "occ_geom_size_head",
                "occ_geom_vis_head",
            ]
        ):
            first_geom_fail = "module_set_available"
            for _m in [
                "occ_geom_patch_norm", "occ_geom_obj_query", "occ_geom_relation",
                "occ_geom_mask_head", "occ_geom_center_head", "occ_geom_size_head", "occ_geom_vis_head"
            ]:
                rlog(f"PARAM_USE step={global_step} module={_m} used=0 reason=missing_module")
            rlog(f"OCC_DECISION step={global_step} fn=geom guard=module_set_available pass=0")
            setattr(self, "_occ_dbg_geom_return_reason", "missing_module")
            setattr(self, "_occ_dbg_used_geom_any", False)
            rlog(f"FIRST_FAIL step={global_step} fn=geom first_fail={first_geom_fail or 'none'}")
            _emit_occ_geom_debug("early_return", "missing_module")
            return torch.zeros((), device=self.device if hasattr(self, "device") else None)
        rlog(f"OCC_DECISION step={global_step} fn=geom guard=module_set_available pass=1")
        if occupancy_aux_outputs is None or video_dict is None:
            if first_geom_fail is None:
                first_geom_fail = "has_inputs"
            for _m in [
                "occ_geom_patch_norm", "occ_geom_obj_query", "occ_geom_relation",
                "occ_geom_mask_head", "occ_geom_center_head", "occ_geom_size_head", "occ_geom_vis_head"
            ]:
                rlog(f"PARAM_USE step={global_step} module={_m} used=0 reason=missing_inputs")
            rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_inputs pass=0")
            setattr(self, "_occ_dbg_geom_return_reason", "missing_inputs")
            setattr(self, "_occ_dbg_used_geom_any", False)
            rlog(f"FIRST_FAIL step={global_step} fn=geom first_fail={first_geom_fail or 'none'}")
            _emit_occ_geom_debug("early_return", "missing_inputs")
            return torch.zeros((), device=self.device if hasattr(self, "device") else None)
        rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_inputs pass=1")

        X = occupancy_aux_outputs.get("patch_embeddings", None)
        E = occupancy_aux_outputs.get("object_embeddings", None)
        obj_ids_union = occupancy_aux_outputs.get("obj_ids_union", None)
        present = occupancy_aux_outputs.get("present", None)
        frame_ids = occupancy_aux_outputs.get("frame_ids", None)

        if X is None or E is None or obj_ids_union is None or present is None:
            if first_geom_fail is None:
                first_geom_fail = "has_aux_tensors"
            for _m in [
                "occ_geom_patch_norm", "occ_geom_obj_query", "occ_geom_relation",
                "occ_geom_mask_head", "occ_geom_center_head", "occ_geom_size_head", "occ_geom_vis_head"
            ]:
                rlog(f"PARAM_USE step={global_step} module={_m} used=0 reason=missing_aux_tensors")
            rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_aux_tensors pass=0")
            setattr(self, "_occ_dbg_geom_return_reason", "missing_aux_tensors")
            setattr(self, "_occ_dbg_used_geom_any", False)
            rlog(f"FIRST_FAIL step={global_step} fn=geom first_fail={first_geom_fail or 'none'}")
            _emit_occ_geom_debug("early_return", "missing_aux_tensors")
            ref = X if X is not None else E
            if ref is None:
                return torch.zeros(())
            return torch.zeros((), device=ref.device, dtype=ref.dtype)
        rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_aux_tensors pass=1")

        device = X.device
        dtype = X.dtype

        patch_occupancy = video_dict.get("patch_occupancy", None)
        visible_bboxes = video_dict.get("visible_bboxes", None)
        if patch_occupancy is None or visible_bboxes is None:
            if first_geom_fail is None:
                first_geom_fail = "has_targets"
            for _m in [
                "occ_geom_patch_norm", "occ_geom_obj_query", "occ_geom_relation",
                "occ_geom_mask_head", "occ_geom_center_head", "occ_geom_size_head", "occ_geom_vis_head"
            ]:
                rlog(f"PARAM_USE step={global_step} module={_m} used=0 reason=missing_targets")
            rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_targets pass=0")
            setattr(self, "_occ_dbg_geom_return_reason", "missing_targets")
            setattr(self, "_occ_dbg_used_geom_any", False)
            rlog(f"FIRST_FAIL step={global_step} fn=geom first_fail={first_geom_fail or 'none'}")
            _emit_occ_geom_debug("early_return", "missing_targets")
            return torch.zeros((), device=device, dtype=dtype)
        rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_targets pass=1")

        T, P, Dp = X.shape
        assert len(patch_occupancy) == T, "patch_occupancy length must match T"
        assert len(visible_bboxes) == T, "visible_bboxes length must match T"
        assert present.shape[:2] == E.shape[:2], "present must align with object_embeddings"

        union_list = [int(v) for v in obj_ids_union.detach().cpu().tolist()]
        union_id_to_col = {oid: i for i, oid in enumerate(union_list)}

        eps = float(getattr(self.config, "occ_geom_eps", 1e-6))
        chunk_size = max(1, int(getattr(self.config, "occ_geom_chunk_size", 8)))
        mask_loss_sum = torch.zeros((), device=device, dtype=dtype)
        box_loss_sum = torch.zeros((), device=device, dtype=dtype)
        ctr_loss_sum = torch.zeros((), device=device, dtype=dtype)
        vis_loss_sum = torch.zeros((), device=device, dtype=dtype)
        mask_bce_loss_sum = torch.zeros((), device=device, dtype=dtype)
        mask_dice_loss_sum = torch.zeros((), device=device, dtype=dtype)
        box_l1_loss_sum = torch.zeros((), device=device, dtype=dtype)
        box_giou_loss_sum = torch.zeros((), device=device, dtype=dtype)
        mask_count = box_count = ctr_count = vis_count = 0
        _geom_use_logged = False
        self._log_occ_cuda_memory(f"geom_start T={T} P={P} O={int(E.shape[1])} Dp={Dp}")

        for f in range(T):
            dbg_stats["frames_total"] += 1
            occ_anno = patch_occupancy[f]
            vis_anno = visible_bboxes[f]
            if occ_anno is None or vis_anno is None:
                continue
            dbg_stats["frames_with_both_annos"] += 1

            if frame_ids is not None and vis_anno.get("frame_id", None) is not None:
                vis_fid = str(vis_anno.get("frame_id"))
                if vis_fid != str(frame_ids[f]):
                    rank0_print(f"[occ_geom] frame_id mismatch at frame {f}: frame_ids={frame_ids[f]}, visible_bboxes={vis_fid}")

            preprocess = occ_anno.get("vision_tower_preprocess", {})
            grid = preprocess.get("grid", None)
            resize = preprocess.get("resize", None)
            if grid is None or resize is None:
                dbg_stats["frames_missing_preprocess"] += 1
                continue
            if int(resize[0]) <= 0 or int(resize[1]) <= 0:
                dbg_stats["frames_missing_preprocess"] += 1
                continue

            src_h, src_w = int(grid[0]), int(grid[1])
            if src_h > 0 and src_w > 0 and (src_h * src_w == P):
                tgt_h, tgt_w = src_h, src_w
            else:
                side = int(math.isqrt(P))
                if side * side != P:
                    dbg_stats["frames_invalid_target_grid"] += 1
                    continue
                tgt_h, tgt_w = side, side
                dbg_stats["frames_grid_mismatch_remapped"] += 1
            dbg_stats["frames_valid_geom_meta"] += 1

            with torch.no_grad():
                patch_centers = self._build_patch_centers_normalized(
                    occ_anno,
                    device=device,
                    dtype=dtype,
                    target_grid=(tgt_h, tgt_w),
                )
                occ_target = self._build_frame_object_occupancy_targets(
                    occ_anno,
                    obj_ids_union,
                    device=device,
                    dtype=dtype,
                    target_grid=(tgt_h, tgt_w),
                )

            union_cols = []
            tgt_boxes = []
            tgt_centers = []
            tgt_center_visible = []

            for vis_obj in vis_anno.get("detected", []):
                dbg_stats["vis_total"] += 1
                if "object_id" not in vis_obj:
                    dbg_stats["vis_missing_object_id"] += 1
                    continue
                oid = int(vis_obj["object_id"])
                if oid not in union_id_to_col:
                    continue
                dbg_stats["vis_in_union"] += 1
                ucol = union_id_to_col[oid]
                if not bool(present[f, ucol].item()):
                    continue
                dbg_stats["vis_present"] += 1

                bbox_norm, center_norm, center_visible = self._convert_visible_bbox_to_geom_target(vis_obj, vis_anno, occ_anno)
                if bbox_norm is None:
                    dbg_stats["vis_bbox_invalid"] += 1
                    continue
                dbg_stats["vis_bbox_valid"] += 1

                union_cols.append(ucol)
                tgt_boxes.append(bbox_norm)
                tgt_centers.append(center_norm)
                tgt_center_visible.append(center_visible)

            if len(union_cols) == 0:
                continue
            dbg_stats["frames_with_candidates"] += 1
            dbg_stats["selected_pairs"] += len(union_cols)

            union_cols_t = torch.tensor(union_cols, device=device, dtype=torch.long)
            x_f = X[f]
            use_occ_geom_patch_norm = bool(getattr(self.config, "use_occ_geom_patch_norm", True))
            use_occ_geom_obj_query = bool(getattr(self.config, "use_occ_geom_obj_query", True))
            if not _geom_use_logged:
                rlog(
                    f"PARAM_USE step={global_step} module=occ_geom_patch_norm "
                    f"used={1 if use_occ_geom_patch_norm else 0}"
                    + ("" if use_occ_geom_patch_norm else " reason=ablation_disabled")
                )
                rlog(
                    f"PARAM_USE step={global_step} module=occ_geom_obj_query "
                    f"used={1 if use_occ_geom_obj_query else 0}"
                    + ("" if use_occ_geom_obj_query else " reason=ablation_disabled")
                )
                for _m in [
                    "occ_geom_relation", "occ_geom_mask_head", "occ_geom_center_head", "occ_geom_size_head", "occ_geom_vis_head"
                ]:
                    rlog(f"PARAM_USE step={global_step} module={_m} used=1")
                _geom_use_logged = True
                used_geom_any = True
            x_feat = self.model.occ_geom_patch_norm(x_f) if use_occ_geom_patch_norm else x_f
            with torch.no_grad():
                target_boxes_t = torch.tensor(tgt_boxes, device=device, dtype=dtype)
                target_vis_t = torch.tensor(
                    [1.0 if flag else 0.0 for flag in tgt_center_visible],
                    device=device,
                    dtype=dtype,
                )

            Ov = union_cols_t.shape[0]
            self._log_occ_cuda_memory(f"geom_frame_start f={f} Ov={Ov}")
            for start in range(0, Ov, chunk_size):
                end = min(start + chunk_size, Ov)
                chunk_cols_t = union_cols_t[start:end]
                e_chunk = E[f, chunk_cols_t, :]
                e_query = self.model.occ_geom_obj_query(e_chunk) if use_occ_geom_obj_query else e_chunk

                x_expand = x_feat.unsqueeze(0).expand(end - start, -1, -1)
                e_expand = e_query.unsqueeze(1).expand(-1, P, -1)
                rel_input = torch.cat([e_expand, x_expand, e_expand * x_expand], dim=-1)
                h = self.model.occ_geom_relation(rel_input)
                if start == 0:
                    self._log_occ_cuda_memory(f"geom_after_first_chunk_relation f={f} Oc={end - start} Ov={Ov}")

                mask_logits = self.model.occ_geom_mask_head(h).squeeze(-1)
                mask_prob = torch.sigmoid(mask_logits)
                alpha = mask_prob / (mask_prob.sum(dim=1, keepdim=True) + eps)
                g = torch.sum(alpha.unsqueeze(-1) * h, dim=1)

                soft_center = torch.sum(alpha.unsqueeze(-1) * patch_centers.unsqueeze(0), dim=1)
                pred_cx = soft_center[:, 0]
                pred_cy = soft_center[:, 1]

                pred_size = torch.sigmoid(self.model.occ_geom_size_head(g))
                pred_w = pred_size[:, 0]
                pred_h = pred_size[:, 1]

                pred_box = torch.stack([
                    pred_cx - 0.5 * pred_w,
                    pred_cy - 0.5 * pred_h,
                    pred_cx + 0.5 * pred_w,
                    pred_cy + 0.5 * pred_h,
                ], dim=-1).clamp(0.0, 1.0)

                pred_vis_logit = self.model.occ_geom_vis_head(g).squeeze(-1)
                center_logprob = F.log_softmax(self.model.occ_geom_center_head(h).squeeze(-1), dim=1)

                for local_k, global_k in enumerate(range(start, end)):
                    ucol = union_cols[global_k]
                    target_mask = occ_target[:, ucol]
                    loss_mask_bce = F.binary_cross_entropy_with_logits(mask_logits[local_k], target_mask, reduction="mean")
                    pred_mask = torch.sigmoid(mask_logits[local_k])
                    dice_num = 2.0 * (pred_mask * target_mask).sum()
                    dice_den = pred_mask.sum() + target_mask.sum() + eps
                    loss_mask_dice = 1.0 - (dice_num / dice_den)
                    loss_mask_obj = loss_mask_bce + float(getattr(self.config, "occ_geom_mask_dice_weight", 0.5)) * loss_mask_dice
                    mask_loss_sum = mask_loss_sum + loss_mask_obj
                    mask_bce_loss_sum = mask_bce_loss_sum + loss_mask_bce
                    mask_dice_loss_sum = mask_dice_loss_sum + loss_mask_dice
                    mask_count += 1

                    target_box = target_boxes_t[global_k].to(dtype=pred_box.dtype)
                    loss_box_l1 = F.smooth_l1_loss(pred_box[local_k], target_box, reduction="mean")
                    loss_box_giou = 1.0 - self._generalized_iou_xyxy(
                        pred_box[local_k].unsqueeze(0),
                        target_box.unsqueeze(0),
                        eps=eps,
                    ).mean()
                    loss_box_obj = loss_box_l1 + float(getattr(self.config, "occ_geom_box_giou_weight", 1.0)) * loss_box_giou
                    box_loss_sum = box_loss_sum + loss_box_obj
                    box_l1_loss_sum = box_l1_loss_sum + loss_box_l1
                    box_giou_loss_sum = box_giou_loss_sum + loss_box_giou
                    box_count += 1

                    target_vis = target_vis_t[global_k].to(dtype=pred_vis_logit.dtype)
                    loss_vis_obj = F.binary_cross_entropy_with_logits(pred_vis_logit[local_k], target_vis, reduction="mean")
                    vis_loss_sum = vis_loss_sum + loss_vis_obj
                    vis_count += 1

                    if tgt_center_visible[global_k] and (tgt_centers[global_k] is not None):
                        with torch.no_grad():
                            cxn, cyn = float(tgt_centers[global_k][0]), float(tgt_centers[global_k][1])
                            dx = patch_centers[:, 0] - cxn
                            dy = patch_centers[:, 1] - cyn
                            bbox_norm = tgt_boxes[global_k]
                            bw = max(float(bbox_norm[2] - bbox_norm[0]), 1e-6)
                            bh = max(float(bbox_norm[3] - bbox_norm[1]), 1e-6)
                            sigma_x = max(1.0 / float(tgt_w), float(getattr(self.config, "occ_geom_center_alpha", 0.1)) * bw)
                            sigma_y = max(1.0 / float(tgt_h), float(getattr(self.config, "occ_geom_center_alpha", 0.1)) * bh)
                            target_ctr = torch.exp(-0.5 * (dx * dx / (sigma_x * sigma_x) + dy * dy / (sigma_y * sigma_y)))
                            target_ctr = target_ctr / (target_ctr.sum() + eps)
                        loss_ctr_obj = F.kl_div(
                            center_logprob[local_k].unsqueeze(0),
                            target_ctr.unsqueeze(0),
                            reduction="batchmean",
                            log_target=False,
                        )
                        ctr_loss_sum = ctr_loss_sum + loss_ctr_obj
                        ctr_count += 1

        if mask_count == 0 and box_count == 0 and ctr_count == 0 and vis_count == 0:
            if first_geom_fail is None:
                first_geom_fail = "has_valid_targets"
            if not _geom_use_logged:
                for _m in [
                    "occ_geom_patch_norm", "occ_geom_obj_query", "occ_geom_relation",
                    "occ_geom_mask_head", "occ_geom_center_head", "occ_geom_size_head", "occ_geom_vis_head"
                ]:
                    rlog(f"PARAM_USE step={global_step} module={_m} used=0 reason=no_valid_targets")
            rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_valid_targets pass=0")
            setattr(self, "_occ_dbg_geom_return_reason", "no_valid_targets")
            setattr(self, "_occ_dbg_used_geom_any", bool(used_geom_any))
            rlog(f"FIRST_FAIL step={global_step} fn=geom first_fail={first_geom_fail or 'none'}")
            _emit_occ_geom_debug("zero_loss", "no_valid_targets")
            return torch.zeros((), device=device, dtype=dtype)
        rlog(f"OCC_DECISION step={global_step} fn=geom guard=has_valid_targets pass=1")

        loss_mask = mask_loss_sum / max(mask_count, 1)
        loss_box = box_loss_sum / max(box_count, 1)
        loss_ctr = ctr_loss_sum / max(ctr_count, 1)
        loss_vis = vis_loss_sum / max(vis_count, 1)
        loss_mask_bce = mask_bce_loss_sum / max(mask_count, 1)
        loss_mask_dice = mask_dice_loss_sum / max(mask_count, 1)
        loss_box_l1 = box_l1_loss_sum / max(box_count, 1)
        loss_box_giou = box_giou_loss_sum / max(box_count, 1)

        geom_loss = (
            float(getattr(self.config, "occ_geom_mask_weight", 1.0)) * loss_mask
            + float(getattr(self.config, "occ_geom_box_weight", 1.0)) * loss_box
            + float(getattr(self.config, "occ_geom_ctr_weight", 1.0)) * loss_ctr
            + float(getattr(self.config, "occ_geom_vis_weight", 1.0)) * loss_vis
        )
        self._log_occ_cuda_memory("geom_end")
        setattr(self, "_occ_geom_mask_loss", loss_mask.float())
        setattr(self, "_occ_geom_box_loss", loss_box.float())
        setattr(self, "_occ_geom_ctr_loss", loss_ctr.float())
        setattr(self, "_occ_geom_vis_loss", loss_vis.float())
        setattr(self, "_occ_geom_mask_bce_loss", loss_mask_bce.float())
        setattr(self, "_occ_geom_mask_dice_loss", loss_mask_dice.float())
        setattr(self, "_occ_geom_box_l1_loss", loss_box_l1.float())
        setattr(self, "_occ_geom_box_giou_loss", loss_box_giou.float())
        setattr(self, "_occ_dbg_geom_return_reason", None)
        setattr(self, "_occ_dbg_used_geom_any", bool(used_geom_any))
        rlog(f"FIRST_FAIL step={global_step} fn=geom first_fail={first_geom_fail or 'none'}")
        _emit_occ_geom_debug("ok", None)
        return geom_loss.float()

    @torch._dynamo.disable
    def compute_occupancy_obj3d_loss(
        self,
        occupancy_aux_outputs: Dict[str, Any],
        video_dict: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = "NA",
    ) -> torch.Tensor:
        setattr(self, "_occ_obj3d_center_loss", None)
        setattr(self, "_occ_obj3d_size_loss", None)

        obj3d_warn_enabled = bool(getattr(self.config, "occ_obj3d_warn", False))

        def _obj3d_warn(msg: str) -> None:
            if obj3d_warn_enabled:
                warnings.warn(msg)
        if occupancy_aux_outputs is None:
            _obj3d_warn("[occ_obj3d_loss] Missing occupancy_aux_outputs; skipping loss.")
            return torch.zeros(())
        if video_dict is None:
            _obj3d_warn("[occ_obj3d_loss] Missing video_dict; skipping loss.")
            ref = occupancy_aux_outputs.get("object_embeddings", None)
            if torch.is_tensor(ref):
                return torch.zeros((), device=ref.device, dtype=ref.dtype)
            return torch.zeros(())
        if not hasattr(self.model, "occ_obj3d_head_shared"):
            _obj3d_warn("[occ_obj3d_loss] Missing occ_obj3d heads; skipping loss.")
            ref = occupancy_aux_outputs.get("object_embeddings", None)
            if torch.is_tensor(ref):
                return torch.zeros((), device=ref.device, dtype=ref.dtype)
            return torch.zeros(())

        Z = occupancy_aux_outputs.get("object_embeddings", None)
        present = occupancy_aux_outputs.get("present", None)
        obj_ids_union = occupancy_aux_outputs.get("obj_ids_union", None)
        obj_labels_union = occupancy_aux_outputs.get("obj_labels_union", None)
        if (Z is None) or (present is None) or (obj_ids_union is None):
            _obj3d_warn("[occ_obj3d_loss] Missing object embeddings/present/object ids; skipping loss.")
            ref = Z if torch.is_tensor(Z) else present
            if torch.is_tensor(ref):
                return torch.zeros((), device=ref.device, dtype=ref.dtype)
            return torch.zeros(())
        if Z.ndim != 3 or present.ndim != 2:
            _obj3d_warn(
                f"[occ_obj3d_loss] Invalid tensor ranks: object_embeddings.ndim={getattr(Z, 'ndim', None)}, "
                f"present.ndim={getattr(present, 'ndim', None)}; skipping loss."
            )
            return torch.zeros((), device=Z.device, dtype=Z.dtype)
        if Z.shape[:2] != present.shape:
            _obj3d_warn(
                f"[occ_obj3d_loss] Shape mismatch between object_embeddings {tuple(Z.shape)} and present {tuple(present.shape)}; "
                "skipping loss."
            )
            return torch.zeros((), device=Z.device, dtype=Z.dtype)
        if obj_ids_union.ndim != 1 or obj_ids_union.shape[0] != Z.shape[1]:
            _obj3d_warn(
                f"[occ_obj3d_loss] obj_ids_union shape mismatch: obj_ids_union={tuple(obj_ids_union.shape)} vs O={Z.shape[1]}; "
                "skipping loss."
            )
            return torch.zeros((), device=Z.device, dtype=Z.dtype)

        scene_id_from_aux = occupancy_aux_outputs.get("scene_id", None)
        scene_id_from_video = video_dict.get("scene_id", None) if isinstance(video_dict, dict) else None
        scene_id = scene_id_from_aux if scene_id_from_aux is not None else scene_id_from_video
        scene_id = str(scene_id) if scene_id is not None else None
        if not scene_id:
            _obj3d_warn("[occ_obj3d_loss] Missing scene_id in aux/video metadata; skipping loss.")
            return torch.zeros((), device=Z.device, dtype=Z.dtype)
        if (scene_id_from_aux is not None) and (scene_id_from_video is not None) and (str(scene_id_from_aux) != str(scene_id_from_video)):
            _obj3d_warn(
                f"[occ_obj3d_loss] scene_id mismatch between aux ({scene_id_from_aux}) and video_dict ({scene_id_from_video})."
            )

        scene_obj3d_map = video_dict.get("obj3d_annotations", None)
        if not isinstance(scene_obj3d_map, dict):
            _obj3d_warn(f"[occ_obj3d_loss] Missing scene-level obj3d annotations for scene={scene_id}; skipping loss.")
            return torch.zeros((), device=Z.device, dtype=Z.dtype)

        matched_embeddings = []
        gt_centers = []
        gt_log_sizes = []
        T, O, D = Z.shape
        for o in range(O):
            oid = int(obj_ids_union[o].item())
            if oid not in scene_obj3d_map:
                _obj3d_warn(f"[occ_obj3d_loss] Missing GT bbox for scene={scene_id}, object_id={oid}")
                continue

            gt_entry = scene_obj3d_map.get(oid, {})
            bbox = gt_entry.get("bbox", None) if isinstance(gt_entry, dict) else None
            center_fallback = gt_entry.get("center", None) if isinstance(gt_entry, dict) else None
            gt_label = str(gt_entry.get("object_label", "")).strip().lower() if isinstance(gt_entry, dict) else ""
            pred_label = ""
            if isinstance(obj_labels_union, list) and (o < len(obj_labels_union)):
                pred_label = str(obj_labels_union[o]).strip().lower()
            if gt_label and pred_label and (gt_label != pred_label):
                _obj3d_warn(
                    f"[occ_obj3d_loss] Label mismatch for scene={scene_id}, object_id={oid}: "
                    f"aux_label={pred_label}, gt_label={gt_label}"
                )

            if (not isinstance(bbox, (list, tuple))) or (len(bbox) != 6):
                _obj3d_warn(
                    f"[occ_obj3d_loss] Invalid bbox for scene={scene_id}, object_id={oid}; "
                    f"expected length 6, got {None if bbox is None else len(bbox)}"
                )
                if isinstance(center_fallback, (list, tuple)) and (len(center_fallback) == 3):
                    _obj3d_warn(
                        f"[occ_obj3d_loss] scene={scene_id}, object_id={oid} has center fallback but missing valid bbox; skipping."
                    )
                continue

            present_mask = present[:, o].to(dtype=torch.bool)
            if not bool(present_mask.any().item()):
                _obj3d_warn(
                    f"[occ_obj3d_loss] object embedding exists but object never present for scene={scene_id}, object_id={oid}; skipping."
                )
                continue
            obj_embedding = Z[present_mask, o].mean(dim=0)
            if obj_embedding.shape[0] != D:
                _obj3d_warn(
                    f"[occ_obj3d_loss] Invalid pooled embedding shape for scene={scene_id}, object_id={oid}: "
                    f"{tuple(obj_embedding.shape)}"
                )
                continue

            gt_center = torch.tensor(bbox[0:3], device=Z.device, dtype=Z.dtype)
            gt_size = torch.tensor(bbox[3:6], device=Z.device, dtype=Z.dtype)
            if gt_center.shape[0] != 3 or gt_size.shape[0] != 3:
                _obj3d_warn(f"[occ_obj3d_loss] Invalid center/size vector for scene={scene_id}, object_id={oid}; skipping.")
                continue
            gt_log_size = torch.log(gt_size.clamp_min(1e-6))

            matched_embeddings.append(obj_embedding)
            gt_centers.append(gt_center)
            gt_log_sizes.append(gt_log_size)

        if len(matched_embeddings) == 0:
            _obj3d_warn(f"[occ_obj3d_loss] No valid matched objects for scene={scene_id}; skipping loss.")
            return torch.zeros((), device=Z.device, dtype=Z.dtype)

        matched_embeddings = torch.stack(matched_embeddings, dim=0)
        gt_center = torch.stack(gt_centers, dim=0)
        gt_log_size = torch.stack(gt_log_sizes, dim=0)

        shared_feat = self.model.occ_obj3d_head_shared(matched_embeddings)
        pred_center_3d = self.model.occ_obj3d_center_head(shared_feat)
        pred_log_size_3d = self.model.occ_obj3d_size_head(shared_feat)

        if pred_center_3d.shape != gt_center.shape:
            _obj3d_warn(
                f"[occ_obj3d_loss] Shape mismatch for center: pred={tuple(pred_center_3d.shape)}, gt={tuple(gt_center.shape)}; "
                "skipping loss."
            )
            return torch.zeros((), device=Z.device, dtype=Z.dtype)
        if pred_log_size_3d.shape != gt_log_size.shape:
            _obj3d_warn(
                f"[occ_obj3d_loss] Shape mismatch for size: pred={tuple(pred_log_size_3d.shape)}, gt={tuple(gt_log_size.shape)}; "
                "skipping loss."
            )
            return torch.zeros((), device=Z.device, dtype=Z.dtype)

        loss_center = F.smooth_l1_loss(pred_center_3d, gt_center, reduction="mean")
        loss_size = F.smooth_l1_loss(pred_log_size_3d, gt_log_size, reduction="mean")
        setattr(self, "_occ_obj3d_center_loss", loss_center.float())
        setattr(self, "_occ_obj3d_size_loss", loss_size.float())
        lambda_center = float(getattr(self.config, "occ_obj3d_center_weight", 1.0))
        lambda_size = float(getattr(self.config, "occ_obj3d_size_weight", 1.0))
        return (lambda_center * loss_center + lambda_size * loss_size).float()

    @torch._dynamo.disable
    def compute_occupancy_temporal_loss(
        self,
        occupancy_aux_outputs: Dict[str, Any],
        global_step: Optional[int] = "NA",
    ) -> torch.Tensor:
        first_temp_fail = None
        used_temp_proj = False
        if not hasattr(self.model, "occ_temp_projector"):
            first_temp_fail = "has_module"
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=missing_module")
            rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_module pass=0")
            setattr(self, "_occ_dbg_temp_return_reason", "missing_module")
            setattr(self, "_occ_dbg_used_temp_proj", False)
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            return torch.zeros(())
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_module pass=1")
        if occupancy_aux_outputs is None:
            if first_temp_fail is None:
                first_temp_fail = "has_inputs"
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=missing_inputs")
            rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_inputs pass=0")
            setattr(self, "_occ_dbg_temp_return_reason", "missing_inputs")
            setattr(self, "_occ_dbg_used_temp_proj", False)
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            return torch.zeros(())
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_inputs pass=1")

        use_simple_occ_temp_loss = bool(getattr(self.config, "use_simple_occ_temp_loss", False))
        use_positive_only_occ_temp_loss = bool(getattr(self.config, "use_positive_only_occ_temp_loss", False))
        use_softmax_occ_temp_loss = bool(getattr(self.config, "use_softmax_occ_temp_loss", False))
        use_original_occ_temp_loss = (not use_simple_occ_temp_loss) and (not use_positive_only_occ_temp_loss)
        Z = occupancy_aux_outputs.get("object_embeddings", None)
        present = occupancy_aux_outputs.get("present", None)
        obj_cat_ids_union = occupancy_aux_outputs.get("obj_cat_ids_union", None)
        missing_aux = (Z is None or present is None)
        if use_original_occ_temp_loss and (obj_cat_ids_union is None):
            missing_aux = True
        if missing_aux:
            if first_temp_fail is None:
                first_temp_fail = "has_aux_tensors"
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=missing_aux_tensors")
            rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_aux_tensors pass=0")
            setattr(self, "_occ_dbg_temp_return_reason", "missing_aux_tensors")
            setattr(self, "_occ_dbg_used_temp_proj", False)
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            ref = Z if Z is not None else present
            if ref is None:
                return torch.zeros(())
            return torch.zeros((), device=ref.device, dtype=ref.dtype if hasattr(ref, 'dtype') else torch.float32)
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_aux_tensors pass=1")

        device = Z.device
        dtype = Z.dtype
        if Z.shape[:2] != present.shape:
            if first_temp_fail is None:
                first_temp_fail = "shape_match"
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=shape_mismatch")
            rlog(f"OCC_DECISION step={global_step} fn=temp guard=shape_match pass=0")
            setattr(self, "_occ_dbg_temp_return_reason", "shape_mismatch")
            setattr(self, "_occ_dbg_used_temp_proj", False)
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            return torch.zeros((), device=device, dtype=dtype)
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=shape_match pass=1")

        T, O, _ = Z.shape
        if T < 2 or O == 0:
            if first_temp_fail is None:
                first_temp_fail = "has_min_frames_and_objects"
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=insufficient_frames_or_objects")
            rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_min_frames_and_objects pass=0")
            setattr(self, "_occ_dbg_temp_return_reason", "insufficient_frames_or_objects")
            setattr(self, "_occ_dbg_used_temp_proj", False)
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            return torch.zeros((), device=device, dtype=dtype)
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_min_frames_and_objects pass=1")
        if use_original_occ_temp_loss:
            if obj_cat_ids_union.shape[0] != O:
                if first_temp_fail is None:
                    first_temp_fail = "category_shape_match"
                rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=category_shape_mismatch")
                rlog(f"OCC_DECISION step={global_step} fn=temp guard=category_shape_match pass=0")
                setattr(self, "_occ_dbg_temp_return_reason", "category_shape_mismatch")
                setattr(self, "_occ_dbg_used_temp_proj", False)
                rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
                return torch.zeros((), device=device, dtype=dtype)
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=category_shape_match pass=1")

        same_min = float(getattr(self.config, "occ_temp_same_min_margin", 0.10))
        same_max = float(getattr(self.config, "occ_temp_same_max_margin", 0.25))
        diff_margin = float(getattr(self.config, "occ_temp_diff_margin", 0.30))
        if use_softmax_occ_temp_loss:
            softmax_tau = float(getattr(self.config, "occ_temp_softmax_tau", 0.07))
            same_neg_weight = float(getattr(self.config, "occ_temp_same_neg_weight", 0.5))
            diff_neg_weight = float(getattr(self.config, "occ_temp_diff_neg_weight", 1.0))
            if not (softmax_tau > 0.0 and same_neg_weight >= 0.0 and diff_neg_weight >= 0.0):
                if first_temp_fail is None:
                    first_temp_fail = "softmax_config_valid"
                rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=invalid_softmax_hparams")
                rlog(f"OCC_DECISION step={global_step} fn=temp guard=softmax_config_valid pass=0")
                setattr(self, "_occ_dbg_temp_return_reason", "invalid_softmax_hparams")
                setattr(self, "_occ_dbg_used_temp_proj", False)
                rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
                return torch.zeros((), device=device, dtype=dtype)
        else:
            if use_original_occ_temp_loss:
                if not (diff_margin > same_max > same_min > 0.0):
                    if first_temp_fail is None:
                        first_temp_fail = "margin_config_valid"
                    rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=invalid_margins")
                    rlog(f"OCC_DECISION step={global_step} fn=temp guard=margin_config_valid pass=0")
                    setattr(self, "_occ_dbg_temp_return_reason", "invalid_margins")
                    setattr(self, "_occ_dbg_used_temp_proj", False)
                    rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
                    return torch.zeros((), device=device, dtype=dtype)
            elif use_simple_occ_temp_loss:
                if not (diff_margin >= 0.0):
                    if first_temp_fail is None:
                        first_temp_fail = "margin_config_valid"
                    rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=invalid_margins")
                    rlog(f"OCC_DECISION step={global_step} fn=temp guard=margin_config_valid pass=0")
                    setattr(self, "_occ_dbg_temp_return_reason", "invalid_margins")
                    setattr(self, "_occ_dbg_used_temp_proj", False)
                    rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
                    return torch.zeros((), device=device, dtype=dtype)
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=margin_config_valid pass=1")

        min_frames = int(getattr(self.config, "occ_temp_min_frames", 2))
        eps = float(getattr(self.config, "occ_temp_eps", 1e-6))

        use_occ_temp_proj = bool(getattr(self.config, "use_occ_temp_projector", True))
        if use_occ_temp_proj:
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=1")
            used_temp_proj = True
            U = self.model.occ_temp_projector(Z)
        else:
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=ablation_disabled")
            U = Z
        U = F.normalize(U, dim=-1)
        self._log_occ_cuda_memory(f"temp_after_projector T={T} O={O}")

        counts = present.long().sum(dim=0)
        if counts.shape[0] != O:
            if first_temp_fail is None:
                first_temp_fail = "count_shape_match"
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=count_shape_mismatch")
            setattr(self, "_occ_dbg_temp_return_reason", "count_shape_mismatch")
            setattr(self, "_occ_dbg_used_temp_proj", bool(used_temp_proj))
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            return torch.zeros((), device=device, dtype=dtype)

        valid_anchor_obj = counts >= min_frames
        valid_proto_obj = counts >= 1
        if not valid_anchor_obj.any():
            if first_temp_fail is None:
                first_temp_fail = "has_valid_anchors"
            rlog(f"PARAM_USE step={global_step} module=occ_temp_projector used=0 reason=no_valid_anchors")
            rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_valid_anchors pass=0")
            setattr(self, "_occ_dbg_temp_return_reason", "no_valid_anchors")
            setattr(self, "_occ_dbg_used_temp_proj", bool(used_temp_proj))
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            return torch.zeros((), device=device, dtype=dtype)
        rlog(f"OCC_DECISION step={global_step} fn=temp guard=has_valid_anchors pass=1")

        present_f = present.unsqueeze(-1).to(U.dtype)
        sum_u = (U * present_f).sum(dim=0)
        proto_all = sum_u / counts.clamp(min=1).to(U.dtype).unsqueeze(-1)
        proto_all = F.normalize(proto_all, dim=-1)
        self._log_occ_cuda_memory("temp_after_proto")

        loss_sum = torch.zeros((), device=device, dtype=dtype)
        anchor_count = 0

        if use_original_occ_temp_loss:
            cat_ids = obj_cat_ids_union.to(device=device)

        for t in range(T):
            for o in range(O):
                if not bool(present[t, o].item()):
                    continue
                if not bool(valid_anchor_obj[o].item()):
                    continue

                pos_count = int((counts[o] - 1).item())
                if pos_count <= 0:
                    continue

                u_anchor = U[t, o]
                pos_sum = sum_u[o] - U[t, o]
                pos_proto = pos_sum / float(max(pos_count, 1))
                pos_proto = F.normalize(pos_proto, dim=-1)
                s_pos = torch.sum(u_anchor * pos_proto, dim=-1)
                loss_pos = 1.0 - s_pos

                if use_positive_only_occ_temp_loss:
                    loss_anchor = float(getattr(self.config, "occ_temp_pos_weight", 1.0)) * loss_pos
                else:
                    is_other = torch.arange(O, device=device) != o
                    valid_neg = is_other & valid_proto_obj.to(device=device)

                    if use_softmax_occ_temp_loss:
                        if use_simple_occ_temp_loss:
                            if valid_neg.any():
                                s_neg = torch.matmul(proto_all[valid_neg], u_anchor)
                                logits = torch.cat([s_pos.unsqueeze(0), s_neg], dim=0) / softmax_tau
                                loss_anchor = -F.log_softmax(logits, dim=0)[0]
                            else:
                                loss_anchor = float(getattr(self.config, "occ_temp_pos_weight", 1.0)) * loss_pos
                        else:
                            same_mask = valid_neg & (cat_ids == cat_ids[o])
                            diff_mask = valid_neg & (cat_ids != cat_ids[o])

                            has_same = bool(same_mask.any().item())
                            has_diff = bool(diff_mask.any().item())
                            if not (has_same or has_diff):
                                loss_anchor = float(getattr(self.config, "occ_temp_pos_weight", 1.0)) * loss_pos
                            else:
                                log_terms = [s_pos / softmax_tau]
                                if has_same and same_neg_weight > 0.0:
                                    s_same = torch.matmul(proto_all[same_mask], u_anchor)
                                    log_terms.append(math.log(same_neg_weight) + torch.logsumexp(s_same / softmax_tau, dim=0))
                                if has_diff and diff_neg_weight > 0.0:
                                    s_diff = torch.matmul(proto_all[diff_mask], u_anchor)
                                    log_terms.append(math.log(diff_neg_weight) + torch.logsumexp(s_diff / softmax_tau, dim=0))
                                log_denom = torch.logsumexp(torch.stack(log_terms), dim=0)
                                loss_anchor = log_denom - (s_pos / softmax_tau)
                    elif use_simple_occ_temp_loss:
                        loss_neg = torch.zeros((), device=device, dtype=dtype)
                        if valid_neg.any():
                            s_neg = torch.matmul(proto_all[valid_neg], u_anchor)
                            gap_neg = s_pos - s_neg
                            loss_neg_i = F.relu(diff_margin - gap_neg)
                            loss_neg = loss_neg_i.mean()
                        loss_anchor = (
                            float(getattr(self.config, "occ_temp_pos_weight", 1.0)) * loss_pos
                            + float(getattr(self.config, "occ_temp_diff_weight", 1.0)) * loss_neg
                        )
                    else:
                        same_mask = valid_neg & (cat_ids == cat_ids[o])
                        diff_mask = valid_neg & (cat_ids != cat_ids[o])

                        loss_same = torch.zeros((), device=device, dtype=dtype)
                        if same_mask.any():
                            s_same = torch.matmul(proto_all[same_mask], u_anchor)
                            gap_same = s_pos - s_same
                            loss_same_i = F.relu(same_min - gap_same) + F.relu(gap_same - same_max)
                            loss_same = loss_same_i.mean()

                        loss_diff = torch.zeros((), device=device, dtype=dtype)
                        if diff_mask.any():
                            s_diff = torch.matmul(proto_all[diff_mask], u_anchor)
                            gap_diff = s_pos - s_diff
                            loss_diff_i = F.relu(diff_margin - gap_diff)
                            loss_diff = loss_diff_i.mean()

                        loss_anchor = (
                            float(getattr(self.config, "occ_temp_pos_weight", 1.0)) * loss_pos
                            + float(getattr(self.config, "occ_temp_same_weight", 1.0)) * loss_same
                            + float(getattr(self.config, "occ_temp_diff_weight", 1.0)) * loss_diff
                        )
                loss_sum = loss_sum + loss_anchor
                anchor_count += 1

        if anchor_count == 0:
            if first_temp_fail is None:
                first_temp_fail = "anchor_count_nonzero"
            setattr(self, "_occ_dbg_temp_return_reason", "anchor_count_zero")
            setattr(self, "_occ_dbg_used_temp_proj", bool(used_temp_proj))
            rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
            return torch.zeros((), device=device, dtype=dtype)
        self._log_occ_cuda_memory(f"temp_end anchor_count={anchor_count}")
        setattr(self, "_occ_dbg_temp_return_reason", None)
        setattr(self, "_occ_dbg_used_temp_proj", bool(used_temp_proj))
        rlog(f"FIRST_FAIL step={global_step} fn=temp first_fail={first_temp_fail or 'none'}")
        return (loss_sum / float(anchor_count)).float()

    def compute_vm_loss(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        mask: torch.Tensor,
    ):
        def _log_vm_cuda_memory(tag: str) -> None:
            if not getattr(self.config, "cycle_debug_memory", False):
                return
            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            rank0_print(
                "[vm_loss][cuda_mem] "
                f"{tag}: allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max_allocated={max_alloc:.2f}GB"
            )

        batch_size = hidden_states.shape[0]
        assert batch_size == 1 and len(images) == 1
        images = images[0]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        num_frames = boi_ids.shape[0]
        patch_h = math.ceil(math.sqrt(self.model.image_embed_len))
        image_hidden_states = torch.zeros((num_frames, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for frame_index, (cur_boi_id, cur_eoi_id) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                # need to remove image_newline tokens
                # rank0_print(cur_boi_id, cur_eoi_id, newline_ids[frame_index * patch_h : (frame_index + 1) * patch_h - 1])

                cur_hidden_states = [hidden_states[0][cur_boi_id : newline_ids[frame_index * patch_h]]]
                for k in range(frame_index * patch_h + 1, (frame_index + 1) * patch_h):
                    cur_hidden_states.append(
                        hidden_states[0][newline_ids[k - 1] + 1 : newline_ids[k]]
                    )
                cur_hidden_states.append(hidden_states[0][newline_ids[(frame_index + 1) * patch_h - 1] + 1 : cur_eoi_id])
                image_hidden_states[frame_index] = torch.cat(cur_hidden_states)

        _log_vm_cuda_memory("after_hidden_states")

        images_std = torch.tensor(self.config.image_std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_mean = torch.tensor(self.config.image_mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_vae = ((images * images_std + images_mean - 0.5) / 0.5).clamp(-1., 1.)
        images_vae = nn.functional.interpolate(images_vae, size=(self.config.decode_image_size, self.config.decode_image_size), mode='bilinear')

        if mask.sum() > 0:
            # recon only masked images
            images_vae = images_vae[mask.bool()]
            image_hidden_states = image_hidden_states[~mask.bool()]
            repeat_factor = int(num_frames // mask.sum().item())
        else:
            return torch.tensor(0.)

        with torch.no_grad():
            posterior = self.model.mm_pixel_decoder.encode(images_vae).latent_dist
            z_q = (
                self._sample_posterior_latents(posterior) - self.model.mm_pixel_decoder.shift_factor
            ) * self.model.mm_pixel_decoder.scaling_factor
            z_q = self._pack_vae_latents(z_q)

        _log_vm_cuda_memory("after_latents")

        with torch.amp.autocast('cuda', dtype=torch.float32):
            # image_hidden_states = self.model.mm_inv_projector.ln_pre(
            #     image_hidden_states) + self.model.mm_inv_projector.pos_embed
            image_hidden_states = self.model.mm_inv_projector.ln_pre(image_hidden_states)
            h = w = int(image_hidden_states.shape[1] ** 0.5)
            image_hidden_states = image_hidden_states.transpose(1, 2).reshape(image_hidden_states.shape[0], image_hidden_states.shape[2], h, w).contiguous()
            vm_loss = self.model.mm_inv_projector(
                z=image_hidden_states.repeat(repeat_factor, 1, 1, 1).contiguous().float(),
                target=z_q.repeat(repeat_factor, 1, 1, 1).contiguous().float(),
                bev=False,
            )
        vm_loss = vm_loss.float().mean()
        _log_vm_cuda_memory("after_vm_loss")
        return vm_loss


    def compute_vm_loss_v2(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        mask: torch.Tensor,
    ):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1 and len(images) == 1
        images = images[0]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        num_frames = boi_ids.shape[0]
        patch_h = math.ceil(math.sqrt(self.model.image_embed_len))
        image_hidden_states = torch.zeros((num_frames, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for frame_index, (cur_boi_id, cur_eoi_id) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                # need to remove image_newline tokens
                # rank0_print(cur_boi_id, cur_eoi_id, newline_ids[frame_index * patch_h : (frame_index + 1) * patch_h - 1])

                cur_hidden_states = [hidden_states[0][cur_boi_id : newline_ids[frame_index * patch_h]]]
                for k in range(frame_index * patch_h + 1, (frame_index + 1) * patch_h):
                    cur_hidden_states.append(
                        hidden_states[0][newline_ids[k - 1] + 1 : newline_ids[k]]
                    )
                cur_hidden_states.append(hidden_states[0][newline_ids[(frame_index + 1) * patch_h - 1] + 1 : cur_eoi_id])
                image_hidden_states[frame_index] = torch.cat(cur_hidden_states)

        images_std = torch.tensor(self.config.image_std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_mean = torch.tensor(self.config.image_mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_vae = ((images * images_std + images_mean - 0.5) / 0.5).clamp(-1., 1.)
        images_vae = nn.functional.interpolate(images_vae, size=(self.config.decode_image_size, self.config.decode_image_size), mode='bilinear')

        with torch.no_grad():
            posterior = self.model.mm_pixel_decoder.encode(images_vae).latent_dist
            z_q = (
                self._sample_posterior_latents(posterior) - self.model.mm_pixel_decoder.shift_factor
            ) * self.model.mm_pixel_decoder.scaling_factor
            z_q = self._pack_vae_latents(z_q)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            # image_hidden_states = self.model.mm_inv_projector.ln_pre(
            #     image_hidden_states) + self.model.mm_inv_projector.pos_embed
            image_hidden_states = self.model.mm_inv_projector.ln_pre(image_hidden_states)
            h = w = int(image_hidden_states.shape[1] ** 0.5)
            image_hidden_states = image_hidden_states.transpose(1, 2).reshape(image_hidden_states.shape[0], image_hidden_states.shape[2], h, w).contiguous()
            vm_loss = self.model.mm_inv_projector(
                z=image_hidden_states.float(),
                x0=z_q.float(),
            )
        vm_loss = vm_loss.float().mean()
        return vm_loss

    def compute_vm_loss_bev(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        bev_image: torch.Tensor,
        mask: torch.Tensor,
    ):
        def _log_bev_cuda_memory(tag: str) -> None:
            if not getattr(self.config, "cycle_debug_memory", False):
                return
            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            rank0_print(
                "[bev_loss][cuda_mem] "
                f"{tag}: allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max_allocated={max_alloc:.2f}GB"
            )

        batch_size = hidden_states.shape[0]
        assert batch_size == 1 and len(images) == 1
        images = images[0]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        num_frames = boi_ids.shape[0]
        patch_h = math.ceil(math.sqrt(self.model.image_embed_len))
        image_hidden_states = torch.zeros((num_frames, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for frame_index, (cur_boi_id, cur_eoi_id) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                # need to remove image_newline tokens
                # rank0_print(cur_boi_id, cur_eoi_id, newline_ids[frame_index * patch_h : (frame_index + 1) * patch_h - 1])

                cur_hidden_states = [hidden_states[0][cur_boi_id : newline_ids[frame_index * patch_h]]]
                for k in range(frame_index * patch_h + 1, (frame_index + 1) * patch_h):
                    cur_hidden_states.append(
                        hidden_states[0][newline_ids[k - 1] + 1 : newline_ids[k]]
                    )
                cur_hidden_states.append(hidden_states[0][newline_ids[(frame_index + 1) * patch_h - 1] + 1 : cur_eoi_id])
                image_hidden_states[frame_index] = torch.cat(cur_hidden_states)

        _log_bev_cuda_memory("after_hidden_states")

        images_vae = ((bev_image - 0.5) / 0.5).clamp(-1., 1.)

        if mask.sum() > 0:
            # take only [unmasked] hidden states as conditions
            image_hidden_states = image_hidden_states[~mask.bool()]

        with torch.no_grad():
            posterior = self.model.mm_pixel_decoder.encode(images_vae).latent_dist
            z_q = (
                self._sample_posterior_latents(posterior) - self.model.mm_pixel_decoder.shift_factor
            ) * self.model.mm_pixel_decoder.scaling_factor
            z_q = self._pack_vae_latents(z_q)
            # filter
            bev_downsample = torch.nn.functional.interpolate(bev_image, size=(z_q.shape[-2], z_q.shape[-1]), mode='bilinear').mean(dim=1)
            loss_mask = (bev_downsample.unsqueeze(1) > 0).bool().repeat(1, z_q.shape[1], 1, 1)

        _log_bev_cuda_memory("after_latents")

        with torch.amp.autocast('cuda', dtype=torch.float32):
            # image_hidden_states = self.model.mm_inv_projector.ln_pre(
            #     image_hidden_states) + self.model.mm_inv_projector.pos_embed
            image_hidden_states = self.model.mm_inv_projector.ln_pre(image_hidden_states)
            h = w = int(image_hidden_states.shape[1] ** 0.5)
            image_hidden_states = image_hidden_states.transpose(1, 2).reshape(image_hidden_states.shape[0], image_hidden_states.shape[2], h, w).contiguous()
            vm_loss = self.model.mm_inv_projector(
                z=image_hidden_states.float(),
                target=z_q.float(),
                bev=True,
            )
        vm_loss = (vm_loss.float() * loss_mask).sum() / loss_mask.sum()
        # vm_loss = vm_loss.float().mean()
        _log_bev_cuda_memory("after_bev_loss")
        return vm_loss


    # Hanwliu
    @torch._dynamo.disable
    def compute_cycle_consistency_loss(
        self,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        video_dict: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        num_walks: Optional[int] = None,
        temperature_app: Union[float, torch.Tensor] = 0.07,
        temperature_geo: float = 0.10,
        geo_sigma: Optional[float] = None,
        topk: Optional[int] = 32,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        3D-aware CRW cycle-consistency loss.

        - appearance similarity: cosine(feat_t, feat_{t+1})
        - geometry similarity:  -||x_t - x_{t+1}||^2 / (2*sigma^2)
        - optional top-k sparsification per row for stability
        """
        def _log_cycle_cuda_memory(tag: str) -> None:
            if not getattr(self.config, "verbose_logging", False):
                return
            if not getattr(self.config, "cycle_debug_memory", False):
                return
            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            rank0_print(
                "[cycle_consistency_loss][cuda_mem] "
                f"{tag}: allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max_allocated={max_alloc:.2f}GB"
            )

        B = hidden_states.shape[0]
        assert B == 1, "Cycle consistency implemented for batch_size==1"

        # ---- 1) Reconstruct per-frame patch features [T, P, D]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        T = boi_ids.shape[0]
        P = self.model.image_embed_len
        D = hidden_states.shape[-1]
        patch_h = math.ceil(math.sqrt(P))  # should be 14

        feats = torch.zeros((T, P, D), dtype=hidden_states.dtype, device=hidden_states.device)

        for t, (cur_boi, cur_eoi) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi is None) or (cur_eoi is None):
                continue

            # remove newline tokens exactly like compute_vm_loss
            cur_rows = [hidden_states[0][cur_boi: newline_ids[t * patch_h]]]
            for k in range(t * patch_h + 1, (t + 1) * patch_h):
                cur_rows.append(hidden_states[0][newline_ids[k - 1] + 1: newline_ids[k]])
            cur_rows.append(hidden_states[0][newline_ids[(t + 1) * patch_h - 1] + 1: cur_eoi])

            feats[t] = torch.cat(cur_rows, dim=0)

        # ---- 2) Visible frames selection
        frame_mask = torch.ones(T, dtype=torch.bool, device=hidden_states.device)
        if mask is not None:
            frame_mask = frame_mask & (~mask.bool())
        visible_idx = frame_mask.nonzero(as_tuple=False).flatten()
        if getattr(self.config, "verbose_logging", False):
            rank0_print(
                "[cycle_consistency_loss] frame_selection "
                f"T={T}, "
                f"mask_present={mask is not None}, "
                f"mask_shape={(tuple(mask.shape) if mask is not None else None)}, "
                f"mask_true_count={(int(mask.sum().item()) if mask is not None else None)}, "
                f"visible_idx={visible_idx.tolist()}"
            )

        if visible_idx.numel() < 2:
            if getattr(self.config, "verbose_logging", False):
                rank0_print(
                    "[cycle_consistency_loss] Disabled: fewer than 2 visible frames. "
                    f"visible_idx={visible_idx.tolist()}, T={T}, mask_present={mask is not None}."
                )
            return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        if (num_walks is None) or (num_walks >= visible_idx.numel()):
            sel = visible_idx
        else:
            perm = torch.randperm(visible_idx.numel(), device=hidden_states.device)
            sel = visible_idx[perm[:num_walks]]
            sel, _ = torch.sort(sel)
        if getattr(self.config, "verbose_logging", False):
            rank0_print(
                "[cycle_consistency_loss] frame_selection_after "
                f"S={sel.numel()}, "
                f"num_walks={num_walks}, "
                f"sel={sel.tolist()}"
            )

        feats = feats[sel]  # [S, P, D]
        S = feats.shape[0]
        if S < 2:
            if getattr(self.config, "verbose_logging", False):
                rank0_print(
                    "[cycle_consistency_loss] Disabled: selected frames < 2. "
                    f"S={S}, visible_count={visible_idx.numel()}, num_walks={num_walks}."
                )
            return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        _log_cycle_cuda_memory("after_frame_selection")

        if getattr(self.config, "verbose_logging", False):
            rank0_print(
                "[cycle_consistency_loss] transition_matrix_info "
                f"P={P}, transition_matrix_size=({P}, {P}), selected_frames={S}"
            )

        # ---- 3) Feature source selection + normalization
        feature_source = getattr(self.config, "cycle_feature_source", "llm")
        if feature_source == "inv_projector":
            inv_projector = getattr(self.model, "mm_inv_projector", None)
            if (
                inv_projector is None
                or not hasattr(inv_projector, "ln_pre")
                or not hasattr(inv_projector, "net")
                or not hasattr(inv_projector.net, "z_embedder_view")
            ):
                if getattr(self.config, "verbose_logging", False):
                    rank0_print(
                        "[cycle_consistency_loss] Requested inv_projector features, "
                        "but mm_inv_projector.z_embedder_view is unavailable. Falling back to LLM features."
                    )
            else:
                feats = inv_projector.ln_pre(feats)
                h = w = int(feats.shape[1] ** 0.5)
                feats = rearrange(feats, "b (h w) c -> b c h w", h=h, w=w).contiguous()
                feats = rearrange(feats, "b c h w -> b (h w) c").contiguous()
                feats = inv_projector.net.z_embedder_view(feats)
        # Normalize appearance features
        feats = F.normalize(feats, dim=-1)

        # ---- 4) Build per-patch 3D coords [S, P, 3]
        coords = None
        valid_patch = None
        if (
            (video_dict is not None)
            and ("world_coords" in video_dict)
            and getattr(self.config, "use_3d_coordinate", True)
        ):
            # video_dict["world_coords"] shape is [B, V, H, W, 3] after merge_video_dict
            with torch.no_grad():
                wc = video_dict["world_coords"][0]  # [V, H, W, 3]
                # average_coordinate_in_patch gives [V, 14, 14, 3]
                wc_p = self.average_coordinate_in_patch(wc)  # [V, 14, 14, 3]
                wc_p = wc_p.reshape(wc_p.shape[0], -1, 3)    # [V, 196, 3]
                coords = wc_p[sel].to(device=hidden_states.device, dtype=torch.float32)

                if getattr(self.config, "cycle_filter_positive_depth", True):
                    # invalid depth in your pipeline becomes (0,0,0) often; treat z<=0 as invalid
                    valid_patch = (coords[..., 2] > 0.0)  # [S, P]
                    if (~valid_patch).any():
                        if getattr(self.config, "verbose_logging", False):
                            rank0_print(
                                "[cycle_consistency_loss] Detected non-positive depth patches; "
                                f"invalid_count={(~valid_patch).sum().item()}."
                            )

        # if no coords, fall back to appearance-only (still works)
        use_geo = coords is not None
        _log_cycle_cuda_memory(f"after_coords_build use_geo={use_geo}")

        # choose sigma automatically if not provided
        if use_geo and (geo_sigma is None):
            # robust scale from consecutive frame patch distances
            with torch.no_grad():
                d = coords[1:] - coords[:-1]  # [S-1, P, 3]
                dist = torch.linalg.norm(d, dim=-1)  # [S-1, P]
                if valid_patch is not None:
                    vm = valid_patch[1:] & valid_patch[:-1]
                    dist = dist[vm]
                if dist.numel() > 0:
                    geo_sigma = torch.quantile(dist, 0.5).clamp(min=1e-3).item()
                else:
                    geo_sigma = 0.10

        # ---- 5) Build neighbor transitions
        A_fwd, A_bwd = [], []
        if torch.is_tensor(temperature_app):
            tau_app = temperature_app
        else:
            tau_app = torch.tensor(
                temperature_app,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        tau_geo = float(temperature_geo)

        for step in range(S - 1):
            Fa = feats[step]     # [P, D]
            Fb = feats[step + 1] # [P, D]

            # appearance logits
            logit_app_fwd = (Fa @ Fb.T) / tau_app
            logit_app_bwd = (Fb @ Fa.T) / tau_app
            if step == 0:
                _log_cycle_cuda_memory("after_first_logit_app")

            if use_geo:
                with torch.no_grad():
                    Xa = coords[step]     # [P, 3]
                    Xb = coords[step + 1] # [P, 3]

                    # squared distances [P, P]
                    # (Xa[:,None,:] - Xb[None,:,:])^2 sum
                    diff = Xa[:, None, :] - Xb[None, :, :]
                    geo_mode = getattr(self.config, "cycle_geo_mode", "raw")
                    if geo_mode == "clamped":
                        min_xyz = torch.tensor(self.config.min_xyz_range, device=diff.device, dtype=diff.dtype)
                        max_xyz = torch.tensor(self.config.max_xyz_range, device=diff.device, dtype=diff.dtype)
                        Xa_c = torch.clamp(Xa, min=min_xyz, max=max_xyz)
                        Xb_c = torch.clamp(Xb, min=min_xyz, max=max_xyz)
                        diff = Xa_c[:, None, :] - Xb_c[None, :, :]
                        dist = torch.linalg.norm(diff, dim=-1)
                        d_max = torch.linalg.norm(max_xyz - min_xyz).clamp_min(1e-6)
                        logit_geo = 1.0 - (2.0 * dist / d_max)
                    else:
                        dist2 = (diff * diff).sum(dim=-1)  # [P, P]
                        logit_geo = -dist2 / (2.0 * (geo_sigma ** 2))

                logit_fwd = logit_app_fwd + (logit_geo / tau_geo)
                logit_bwd = logit_app_bwd + (logit_geo.T / tau_geo)

                # mask invalid patches (rows/cols)
                if valid_patch is not None:
                    va = valid_patch[step]     # [P]
                    vb = valid_patch[step + 1] # [P]
                    # invalid source rows -> forbid transitions
                    logit_fwd = logit_fwd.masked_fill(~va[:, None], float("-inf"))
                    logit_bwd = logit_bwd.masked_fill(~vb[:, None], float("-inf"))
                    # invalid target cols -> forbid transitions
                    logit_fwd = logit_fwd.masked_fill(~vb[None, :], float("-inf"))
                    logit_bwd = logit_bwd.masked_fill(~va[None, :], float("-inf"))
                    # if an entire row is invalid, skip cycle loss for this batch
                    if (~torch.isfinite(logit_fwd)).all(dim=-1).any() or (~torch.isfinite(logit_bwd)).all(dim=-1).any():
                        invalid_rows_fwd = (~torch.isfinite(logit_fwd)).all(dim=-1).sum().item()
                        invalid_rows_bwd = (~torch.isfinite(logit_bwd)).all(dim=-1).sum().item()
                        if getattr(self.config, "verbose_logging", False):
                            rank0_print(
                                "[cycle_consistency_loss] Disabled: invalid transition rows after masking. "
                                f"step={step}, invalid_rows_fwd={invalid_rows_fwd}, invalid_rows_bwd={invalid_rows_bwd}, "
                                f"valid_patch_ratio={(valid_patch.float().mean().item() if valid_patch is not None else 'n/a')}."
                            )
                        return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                logit_fwd = logit_app_fwd
                logit_bwd = logit_app_bwd

            # top-k sparsification (helps prevent uniform collapse early)
            if (topk is not None) and (topk < P):
                v, idx = torch.topk(logit_fwd, k=topk, dim=-1)
                sparse = torch.full_like(logit_fwd, float("-inf"))
                sparse.scatter_(-1, idx, v)
                logit_fwd = sparse

                v, idx = torch.topk(logit_bwd, k=topk, dim=-1)
                sparse = torch.full_like(logit_bwd, float("-inf"))
                sparse.scatter_(-1, idx, v)
                logit_bwd = sparse

            A_fwd.append(F.softmax(logit_fwd, dim=-1))
            A_bwd.append(F.softmax(logit_bwd, dim=-1))
            if step == 0:
                _log_cycle_cuda_memory("after_first_transition")

        # ---- 6) Compose forward then backward
        if len(A_fwd) != (S - 1):
            if getattr(self.config, "verbose_logging", False):
                rank0_print(
                    "[cycle_consistency_loss] Disabled: transition list length mismatch. "
                    f"len(A_fwd)={len(A_fwd)}, expected={S - 1}, S={S}."
                )
            return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        A_fw = A_fwd[0]
        for k in range(1, S - 1):
            A_fw = A_fw @ A_fwd[k]   # [P, P]

        A_bw = A_bwd[-1]
        for k in range(S - 3, -1, -1):
            A_bw = A_bw @ A_bwd[k]   # [P, P]

        M = A_fw @ A_bw  # [P, P]
        _log_cycle_cuda_memory("after_cycle_matrix")

        diag = torch.diagonal(M, dim1=-2, dim2=-1)  # [P]

        # if valid_patch exists, only score patches valid in the first selected frame
        if valid_patch is not None:
            v0 = valid_patch[0]
            if v0.any():
                diag = diag[v0]
            else:
                if getattr(self.config, "verbose_logging", False):
                    rank0_print(
                        "[cycle_consistency_loss] Disabled: no valid patches in first selected frame. "
                        f"valid_patch_shape={valid_patch.shape}, selected_frames={sel.tolist()}."
                    )
                return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        loss = -torch.log(diag + eps).mean()
        return loss

    @torch._dynamo.disable
    def compute_cycle_consistency_loss_v2(
        self,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        video_dict: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        num_walks: Optional[int] = None,
        temperature_app: Union[float, torch.Tensor] = 0.07,
        temperature_geo: float = 0.10,
        geo_sigma: Optional[float] = None,
        topk: Optional[int] = 32,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        v2: 3D-aware CRW cycle-consistency loss with *geometry-guided distillation*.

        - appearance logits: (Fa @ Fb^T)  (cosine sim since feats normalized)
        - geometry logits:   -||Xa - Xb||^2 / (2*sigma^2)
        - normalization: masked z-score for app and geo logits so neither dominates
        - guidance loss: KL( q_geo || p_app ) row-wise + column-wise (symmetry)
        """
        def _log_cycle_cuda_memory(tag: str) -> None:
            if not getattr(self.config, "verbose_logging", False):
                return
            if not getattr(self.config, "cycle_debug_memory", False):
                return
            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            rank0_print(
                "[cycle_consistency_loss_v2][cuda_mem] "
                f"{tag}: allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max_allocated={max_alloc:.2f}GB"
            )

        B = hidden_states.shape[0]
        assert B == 1, "Cycle consistency implemented for batch_size==1"

        # ---- 1) Reconstruct per-frame patch features [T, P, D]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        T = boi_ids.shape[0]
        P = self.model.image_embed_len
        D = hidden_states.shape[-1]
        patch_h = math.ceil(math.sqrt(P))  # should be 14

        feats = torch.zeros((T, P, D), dtype=hidden_states.dtype, device=hidden_states.device)

        for t, (cur_boi, cur_eoi) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi is None) or (cur_eoi is None):
                continue

            # remove newline tokens exactly like compute_vm_loss
            cur_rows = [hidden_states[0][cur_boi: newline_ids[t * patch_h]]]
            for k in range(t * patch_h + 1, (t + 1) * patch_h):
                cur_rows.append(hidden_states[0][newline_ids[k - 1] + 1: newline_ids[k]])
            cur_rows.append(hidden_states[0][newline_ids[(t + 1) * patch_h - 1] + 1: cur_eoi])

            feats[t] = torch.cat(cur_rows, dim=0)

        # ---- 2) Visible frames selection
        frame_mask = torch.ones(T, dtype=torch.bool, device=hidden_states.device)
        if mask is not None:
            frame_mask = frame_mask & (~mask.bool())
        visible_idx = frame_mask.nonzero(as_tuple=False).flatten()
        if getattr(self.config, "verbose_logging", False):
            rank0_print(
                "[cycle_consistency_loss_v2] frame_selection "
                f"T={T}, "
                f"mask_present={mask is not None}, "
                f"mask_shape={(tuple(mask.shape) if mask is not None else None)}, "
                f"mask_true_count={(int(mask.sum().item()) if mask is not None else None)}, "
                f"visible_idx={visible_idx.tolist()}"
            )

        if visible_idx.numel() < 2:
            if getattr(self.config, "verbose_logging", False):
                rank0_print(
                    "[cycle_consistency_loss_v2] Disabled: fewer than 2 visible frames. "
                    f"visible_idx={visible_idx.tolist()}, T={T}, mask_present={mask is not None}."
                )
            return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        if (num_walks is None) or (num_walks >= visible_idx.numel()):
            sel = visible_idx
        else:
            perm = torch.randperm(visible_idx.numel(), device=hidden_states.device)
            sel = visible_idx[perm[:num_walks]]
            sel, _ = torch.sort(sel)
        if getattr(self.config, "verbose_logging", False):
            rank0_print(
                "[cycle_consistency_loss_v2] frame_selection_after "
                f"S={sel.numel()}, "
                f"num_walks={num_walks}, "
                f"sel={sel.tolist()}"
            )

        feats = feats[sel]  # [S, P, D]
        S = feats.shape[0]
        if S < 2:
            if getattr(self.config, "verbose_logging", False):
                rank0_print(
                    "[cycle_consistency_loss_v2] Disabled: selected frames < 2. "
                    f"S={S}, visible_count={visible_idx.numel()}, num_walks={num_walks}."
                )
            return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        _log_cycle_cuda_memory("after_frame_selection")

        # ---- 3) Feature source selection + normalization
        feature_source = getattr(self.config, "cycle_feature_source", "llm")
        if feature_source == "inv_projector":
            inv_projector = getattr(self.model, "mm_inv_projector", None)
            if (
                inv_projector is None
                or not hasattr(inv_projector, "ln_pre")
                or not hasattr(inv_projector, "net")
                or not hasattr(inv_projector.net, "z_embedder_view")
            ):
                if getattr(self.config, "verbose_logging", False):
                    rank0_print(
                        "[cycle_consistency_loss_v2] Requested inv_projector features, "
                        "but mm_inv_projector.z_embedder_view is unavailable. Falling back to LLM features."
                    )
            else:
                feats = inv_projector.ln_pre(feats)
                h = w = int(feats.shape[1] ** 0.5)
                feats = rearrange(feats, "b (h w) c -> b c h w", h=h, w=w).contiguous()
                feats = rearrange(feats, "b c h w -> b (h w) c").contiguous()
                feats = inv_projector.net.z_embedder_view(feats)
        feats = F.normalize(feats, dim=-1)  # [S, P, D]

        # ---- 4) Build per-patch 3D coords [S, P, 3] (WORLD coords)
        coords = None
        valid_patch = None
        if (
            (video_dict is not None)
            and ("world_coords" in video_dict)
            and getattr(self.config, "use_3d_coordinate", True)
        ):
            with torch.no_grad():
                wc = video_dict["world_coords"][0]           # [V, H, W, 3]
                wc_p = self.average_coordinate_in_patch(wc)  # [V, 14, 14, 3]
                wc_p = wc_p.reshape(wc_p.shape[0], -1, 3)    # [V, 196, 3]
                coords = wc_p[sel].to(device=hidden_states.device, dtype=torch.float32)

                if getattr(self.config, "cycle_filter_positive_depth", True):
                    valid_patch = (coords[..., 2] > 0.0)  # [S, P]

        use_geo = coords is not None
        _log_cycle_cuda_memory(f"after_coords_build use_geo={use_geo}")

        # choose sigma automatically if not provided (your existing robust heuristic)
        if use_geo and (geo_sigma is None):
            with torch.no_grad():
                d = coords[1:] - coords[:-1]               # [S-1, P, 3]
                dist = torch.linalg.norm(d, dim=-1)        # [S-1, P]
                if valid_patch is not None:
                    vm = valid_patch[1:] & valid_patch[:-1]
                    dist = dist[vm]
                if dist.numel() > 0:
                    geo_sigma = torch.quantile(dist, 0.5).clamp(min=1e-3).item()
                else:
                    geo_sigma = 0.10

        # -------------------------------------------------------------------------
        # v2 core: masked z-score normalization + KL guidance
        # -------------------------------------------------------------------------
        def _masked_zscore(x: torch.Tensor, dim: int, eps_: float) -> torch.Tensor:
            """
            x: [P, P] with possible -inf entries.
            returns z: [P, P] where finite entries are z-scored along `dim`,
            and non-finite entries remain -inf.
            """
            finite = torch.isfinite(x)
            x0 = torch.where(finite, x, torch.zeros_like(x))
            cnt = finite.sum(dim=dim, keepdim=True).clamp_min(1)
            mu = x0.sum(dim=dim, keepdim=True) / cnt
            diff = x0 - mu
            var = torch.where(finite, diff * diff, torch.zeros_like(diff)).sum(dim=dim, keepdim=True) / cnt
            std = torch.sqrt(var + eps_)
            z = diff / std
            z = z.masked_fill(~finite, float("-inf"))
            return z

        def _kl_rowcol(app_logits: torch.Tensor,
                    geo_logits: torch.Tensor,
                    src_valid: Optional[torch.Tensor],
                    tgt_valid: Optional[torch.Tensor],
                    tau_a: float,
                    tau_g: float,
                    eps_: float) -> torch.Tensor:
            """
            app_logits, geo_logits: [P, P] (masked with -inf where invalid)
            src_valid, tgt_valid: [P] boolean or None
            returns scalar KL(row) + KL(col)
            """
            la = app_logits
            lg = geo_logits

            if src_valid is not None:
                la = la.masked_fill(~src_valid[:, None], float("-inf"))
                lg = lg.masked_fill(~src_valid[:, None], float("-inf"))
            if tgt_valid is not None:
                la = la.masked_fill(~tgt_valid[None, :], float("-inf"))
                lg = lg.masked_fill(~tgt_valid[None, :], float("-inf"))

            # If any row is entirely invalid => avoid NaNs in softmax
            if (~torch.isfinite(la)).all(dim=-1).any() or (~torch.isfinite(lg)).all(dim=-1).any():
                return torch.zeros((), device=la.device, dtype=la.dtype)

            # Row distributions
            la_r = _masked_zscore(la, dim=-1, eps_=eps_)
            lg_r = _masked_zscore(lg, dim=-1, eps_=eps_)

            p_row = F.softmax(la_r / tau_a, dim=-1)                 # [P, P]
            q_row = F.softmax(lg_r / tau_g, dim=-1).detach()        # [P, P]

            kl_row = (q_row * (torch.log(q_row + eps_) - torch.log(p_row + eps_))).sum(dim=-1)  # [P]

            row_mask = torch.ones((P,), device=la.device, dtype=torch.bool)
            if src_valid is not None:
                row_mask = row_mask & src_valid
            # also require at least one finite target in both
            row_mask = row_mask & torch.isfinite(la).any(dim=-1) & torch.isfinite(lg).any(dim=-1)
            if row_mask.any():
                kl_row = kl_row[row_mask].mean()
            else:
                kl_row = torch.zeros((), device=la.device, dtype=la.dtype)

            # Column distributions (symmetry)
            if (~torch.isfinite(la)).all(dim=0).any() or (~torch.isfinite(lg)).all(dim=0).any():
                kl_col = torch.zeros((), device=la.device, dtype=la.dtype)
            else:
                la_c = _masked_zscore(la, dim=0, eps_=eps_)
                lg_c = _masked_zscore(lg, dim=0, eps_=eps_)

                p_col = F.softmax(la_c / tau_a, dim=0)              # [P, P]
                q_col = F.softmax(lg_c / tau_g, dim=0).detach()     # [P, P]

                kl_col_vec = (q_col * (torch.log(q_col + eps_) - torch.log(p_col + eps_))).sum(dim=0)  # [P]

                col_mask = torch.ones((P,), device=la.device, dtype=torch.bool)
                if tgt_valid is not None:
                    col_mask = col_mask & tgt_valid
                col_mask = col_mask & torch.isfinite(la).any(dim=0) & torch.isfinite(lg).any(dim=0)
                if col_mask.any():
                    kl_col = kl_col_vec[col_mask].mean()
                else:
                    kl_col = torch.zeros((), device=la.device, dtype=la.dtype)

            return kl_row + kl_col

        # ---- 5) Guidance loss
        if torch.is_tensor(temperature_app):
            tau_app = temperature_app
        else:
            tau_app = torch.tensor(
                temperature_app,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        tau_geo = float(temperature_geo)

        # weight for KL guidance
        kl_w = float(getattr(self.config, "cycle_geo_kl_weight", 1.0))

        guidance_loss = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        for step in range(S - 1):
            Fa = feats[step]     # [P, D]
            Fb = feats[step + 1] # [P, D]

            # appearance logits (NOT yet divided by tau; we z-score first)
            logit_app_fwd = (Fa @ Fb.T)  # [P, P]
            logit_app_bwd = (Fb @ Fa.T)  # [P, P]
            if step == 0:
                _log_cycle_cuda_memory("after_first_logit_app")

            if use_geo:
                with torch.no_grad():
                    Xa = coords[step]        # [P, 3]
                    Xb = coords[step + 1]    # [P, 3]

                    diff = Xa[:, None, :] - Xb[None, :, :]   # [P, P, 3]
                    geo_mode = getattr(self.config, "cycle_geo_mode", "raw")
                    if geo_mode == "clamped":
                        min_xyz = torch.tensor(self.config.min_xyz_range, device=diff.device, dtype=diff.dtype)
                        max_xyz = torch.tensor(self.config.max_xyz_range, device=diff.device, dtype=diff.dtype)
                        Xa_c = torch.clamp(Xa, min=min_xyz, max=max_xyz)
                        Xb_c = torch.clamp(Xb, min=min_xyz, max=max_xyz)
                        diff = Xa_c[:, None, :] - Xb_c[None, :, :]
                        dist = torch.linalg.norm(diff, dim=-1)
                        d_max = torch.linalg.norm(max_xyz - min_xyz).clamp_min(1e-6)
                        logit_geo_fwd = 1.0 - (2.0 * dist / d_max)
                        logit_geo_bwd = logit_geo_fwd.T
                    else:
                        dist2 = (diff * diff).sum(dim=-1)        # [P, P]
                        logit_geo = -dist2 / (2.0 * (geo_sigma ** 2))  # [P, P]
                        logit_geo_fwd = logit_geo
                        logit_geo_bwd = logit_geo.T

                # mask invalid patches (rows/cols)
                if valid_patch is not None:
                    va = valid_patch[step]     # [P]
                    vb = valid_patch[step + 1] # [P]

                    # apply the same masking to app & geo (so z-score/softmax ignore invalid)
                    logit_app_fwd = logit_app_fwd.masked_fill(~va[:, None], float("-inf"))
                    logit_geo_fwd = logit_geo_fwd.masked_fill(~va[:, None], float("-inf"))
                    logit_app_fwd = logit_app_fwd.masked_fill(~vb[None, :], float("-inf"))
                    logit_geo_fwd = logit_geo_fwd.masked_fill(~vb[None, :], float("-inf"))
                    logit_app_bwd = logit_app_bwd.masked_fill(~vb[:, None], float("-inf"))
                    logit_geo_bwd = logit_geo_bwd.masked_fill(~vb[:, None], float("-inf"))
                    logit_app_bwd = logit_app_bwd.masked_fill(~va[None, :], float("-inf"))
                    logit_geo_bwd = logit_geo_bwd.masked_fill(~va[None, :], float("-inf"))

                    # if an entire row is invalid, skip
                    if (
                        (~torch.isfinite(logit_app_fwd)).all(dim=-1).any()
                        or (~torch.isfinite(logit_geo_fwd)).all(dim=-1).any()
                        or (~torch.isfinite(logit_app_bwd)).all(dim=-1).any()
                        or (~torch.isfinite(logit_geo_bwd)).all(dim=-1).any()
                    ):
                        if getattr(self.config, "verbose_logging", False):
                            rank0_print(
                                "[cycle_consistency_loss_v2] Disabled: invalid transition rows after masking. "
                                f"step={step}, valid_patch_ratio={valid_patch.float().mean().item():.4f}."
                            )
                        return torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

                # ---- v2 guidance term: KL( q_geo || p_app ) row + col, forward + backward
                if kl_w > 0.0:
                    if valid_patch is not None:
                        va = valid_patch[step]
                        vb = valid_patch[step + 1]
                    else:
                        va = vb = None

                    g_f = _kl_rowcol(
                        app_logits=logit_app_fwd,
                        geo_logits=logit_geo_fwd,
                        src_valid=va,
                        tgt_valid=vb,
                        tau_a=tau_app,
                        tau_g=tau_geo,
                        eps_=eps,
                    )
                    g_b = _kl_rowcol(
                        app_logits=logit_app_bwd,
                        geo_logits=logit_geo_bwd,
                        src_valid=vb,
                        tgt_valid=va,
                        tau_a=tau_app,
                        tau_g=tau_geo,
                        eps_=eps,
                    )
                    guidance_loss = guidance_loss + 0.5 * (g_f + g_b)

        # average guidance loss over steps
        if use_geo and (S > 1) and (kl_w > 0.0):
            guidance_loss = guidance_loss / float(S - 1)
        else:
            guidance_loss = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        # total loss = CRW cycle loss + KL guidance (weight from config)
        # loss = cycle_loss + kl_w * guidance_loss
        loss = kl_w * guidance_loss
        return loss


@dataclass
class CausalLMOutputWithPastRoss(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    vm_loss: Optional[torch.FloatTensor] = None
    bev_loss: Optional[torch.FloatTensor] = None
    n_tokens: Optional[torch.LongTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    scores: Optional[torch.FloatTensor] = None
    # Hanwliu
    cycle_loss: Optional[torch.FloatTensor] = None
    occupancy_aux_outputs: Optional[Dict[str, Union[torch.Tensor, List[int], List[str], str]]] = None
    occ_geom_loss: Optional[torch.FloatTensor] = None
    occ_temp_loss: Optional[torch.FloatTensor] = None
    occ_geom_mask_loss: Optional[torch.FloatTensor] = None
    occ_geom_box_loss: Optional[torch.FloatTensor] = None
    occ_geom_ctr_loss: Optional[torch.FloatTensor] = None
    occ_geom_vis_loss: Optional[torch.FloatTensor] = None
    occ_geom_mask_bce_loss: Optional[torch.FloatTensor] = None
    occ_geom_mask_dice_loss: Optional[torch.FloatTensor] = None
    occ_geom_box_l1_loss: Optional[torch.FloatTensor] = None
    occ_geom_box_giou_loss: Optional[torch.FloatTensor] = None
    occ_obj3d_loss: Optional[torch.FloatTensor] = None
    occ_obj3d_center_loss: Optional[torch.FloatTensor] = None
    occ_obj3d_size_loss: Optional[torch.FloatTensor] = None
