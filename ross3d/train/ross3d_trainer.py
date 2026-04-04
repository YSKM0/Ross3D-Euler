import os
import inspect
import copy
import torch
import torch.nn as nn
import datetime

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.utils.data import Dataset, Sampler, DataLoader

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    is_accelerate_available,
    is_datasets_available,
    is_xpu_available,
    is_mlu_available,
    is_npu_available,
    is_torch_version,
    is_mps_available,
    OptimizerNames,
)
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.trainer import TRAINER_STATE_NAME
from typing import List, Optional
from datetime import timedelta
from pathlib import Path

from ross3d.utils import rank0_print
def rlog(msg):
    if os.getenv("ROSS3D_RLOG_DEBUG", "0") != "1":
        return
    if torch._dynamo.is_compiling():
        return
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[RANK {rank}] {msg}", flush=True)


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs

if is_datasets_available():
    import datasets

from ross3d.utils import rank0_print


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # assert all(l != 0 for l in lengths), "Should not have zero length."
    # if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
    #     # all samples are in the same modality
    #     return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    try:
        ground_indices, ground_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l == 1])
    except:
        ground_indices, ground_lengths = [], []
    try:
        qa_indices, qa_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l == 2])
    except:
        qa_indices, qa_lengths = [], []
    try:
        cap_indices, cap_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l == 3])
    except:
        cap_indices, cap_lengths = [], []

    ground_shuffle = [ground_indices[i] for i in get_length_grouped_indices(ground_lengths, batch_size, world_size, generator=None)]
    qa_shuffle = [qa_indices[i] for i in get_length_grouped_indices(qa_lengths, batch_size, world_size, generator=None)]
    cap_shuffle = [cap_indices[i] for i in get_length_grouped_indices(cap_lengths, batch_size, world_size, generator=None)]

    megabatch_size = world_size * batch_size
    
    ground_megabatches = [ground_shuffle[i : i + megabatch_size] for i in range(0, len(ground_shuffle), megabatch_size)]
    qa_megabatches = [qa_shuffle[i : i + megabatch_size] for i in range(0, len(qa_shuffle), megabatch_size)]
    cap_megabatches = [cap_shuffle[i : i + megabatch_size] for i in range(0, len(cap_shuffle), megabatch_size)]

    # last_mm = mm_megabatches[-1]
    # last_lang = lang_megabatches[-1]
    # additional_batch = last_mm + last_lang
    megabatches = ground_megabatches[:-1] + qa_megabatches[:-1] + cap_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_task_length_grouped_indices(lengths, batch_size, world_size, generator=None):

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    from collections import defaultdict
    task_indices, task_lengths = defaultdict(list), defaultdict(list)
    for i, (task_id, l) in enumerate(lengths):
        task_indices[task_id].append(i)
        task_lengths[task_id].append(l)
    
    task_ids = list(task_indices.keys())
    task_shuffle = {}
    for task_id in task_ids:
        task_shuffle[task_id] = [task_indices[task_id][i] for i in get_length_grouped_indices(task_lengths[task_id], batch_size, world_size, generator=None)]

    megabatch_size = world_size * batch_size
    task_megabatches = {}
    for task_id in task_ids:
        task_megabatches[task_id] = [task_shuffle[task_id][i: i + megabatch_size] for i in range(0, len(task_shuffle[task_id]), megabatch_size)]

    megabatches = []
    for task_id in task_ids:
        megabatches.extend(task_megabatches[task_id][:-1])
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
        group_by_task: bool=False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto
        self.group_by_task = group_by_task

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_task:
            indices = get_task_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        elif self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class Ross3DTrainer(Trainer):
    def _maybe_install_backward_markers(self) -> None:
        if getattr(self, "_rlog_backward_wrapped", False):
            return
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None or not hasattr(accelerator, "backward"):
            return
        orig_backward = accelerator.backward

        def _wrapped_backward(*args, **kwargs):
            rlog("BEFORE_BACKWARD")
            out = orig_backward(*args, **kwargs)
            rlog("AFTER_BACKWARD")
            return out

        accelerator.backward = _wrapped_backward
        self._rlog_backward_wrapped = True

    def _maybe_install_occ_grad_hooks(self) -> None:
        if getattr(self, "_occ_hook_installed", False):
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        targets = {
            "occupancy_patch_projector": "occ_patch_proj",
            "occupancy_object_norm": "occ_obj_norm",
            "occ_temp_projector": "occ_temp_proj",
        }
        self._occ_hook_fired = {v: 0 for v in targets.values()}
        self._occ_hook_handles = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            for token, short_name in targets.items():
                if token in name and short_name not in getattr(self, "_occ_hook_target_names", {}):
                    if not hasattr(self, "_occ_hook_target_names"):
                        self._occ_hook_target_names = {}
                    self._occ_hook_target_names[short_name] = name

                    def _make_hook(key):
                        def _hook(_grad):
                            self._occ_hook_fired[key] = 1
                            return _grad
                        return _hook

                    self._occ_hook_handles.append(param.register_hook(_make_hook(short_name)))
        self._occ_hook_installed = True

    def _reset_occ_hook_fired(self) -> None:
        if hasattr(self, "_occ_hook_fired"):
            for key in self._occ_hook_fired:
                self._occ_hook_fired[key] = 0

    def _log_occ_hook_summary(self) -> None:
        if os.getenv("ROSS3D_RLOG_DEBUG", "0") != "1":
            return
        step = int(getattr(self.state, "global_step", -1))
        max_steps = int(os.getenv("ROSS3D_OCC_HOOK_STEPS", "20"))
        if step >= max_steps:
            return
        fired = getattr(self, "_occ_hook_fired", {})
        patch = int(fired.get("occ_patch_proj", 0))
        norm = int(fired.get("occ_obj_norm", 0))
        temp = int(fired.get("occ_temp_proj", 0))
        rlog(f"OCC_HOOKS step={step} patch_proj={patch} obj_norm={norm} temp_proj={temp}")

    def _log_suspect_module_grad_presence(self, tag: str) -> None:
        if os.getenv("ROSS3D_RLOG_DEBUG", "0") != "1":
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        step = int(getattr(self.state, "global_step", -1))
        max_steps = int(os.getenv("ROSS3D_GRAD_PRESENCE_STEPS", "10"))
        if step >= max_steps:
            return
        suspects = [
            "occupancy_patch_projector",
            "occupancy_object_norm",
            "occ_temp_projector",
            "mm_inv_projector",
            "mm_projector",
            "vision_resampler",
            "ground_head",
        ]
        status = {}
        for name, param in model.named_parameters():
            for key in suspects:
                if key in name:
                    prev = status.get(key, 0)
                    if param.grad is not None:
                        status[key] = 1
                    elif key not in status:
                        status[key] = prev
        compact = " ".join(f"{k}={status.get(k, 0)}" for k in suspects)
        rlog(f"GRAD_PRESENCE tag={tag} step={step} {compact}")

    def _maybe_init_cycle_grad_debug(self):
        model = getattr(self, "model", None)
        if model is None:
            return
        if not getattr(getattr(model, "config", None), "cycle_debug_grad", False):
            return
        if hasattr(self, "_cycle_grad_hook"):
            return
        self._cycle_grad_seen = False
        self._cycle_grad_param = None
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "model." in name or "lm_head" in name:
                self._cycle_grad_param = name
                self._cycle_grad_hook = param.register_hook(lambda *_: setattr(self, "_cycle_grad_seen", True))
                break
        if self._cycle_grad_param is None:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self._cycle_grad_param = name
                self._cycle_grad_hook = param.register_hook(lambda *_: setattr(self, "_cycle_grad_seen", True))
                break
        if self._cycle_grad_param is not None:
            rank0_print(
                "[cycle_debug] registered grad hook on "
                f"{self._cycle_grad_param}"
            )

    def _log_grad_stats(self, tag: str) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        if not getattr(getattr(model, "config", None), "cycle_debug_grad", False):
            return
        total_params = 0
        total_numel = 0
        grad_params = 0
        grad_numel = 0
        groups = {
            "llm": {"params": 0, "numel": 0},
            "mm_inv_projector": {"params": 0, "numel": 0},
            "mm_projector": {"params": 0, "numel": 0},
            "vision_tower": {"params": 0, "numel": 0},
            "vision_resampler": {"params": 0, "numel": 0},
            "other": {"params": 0, "numel": 0},
        }
        for name, param in model.named_parameters():
            total_params += 1
            total_numel += param.numel()
            has_grad = param.grad is not None
            if has_grad:
                grad_params += 1
                grad_numel += param.numel()
            if "mm_inv_projector" in name:
                key = "mm_inv_projector"
            elif "mm_projector" in name:
                key = "mm_projector"
            elif "vision_tower" in name:
                key = "vision_tower"
            elif "vision_resampler" in name:
                key = "vision_resampler"
            elif "model." in name or "lm_head" in name:
                key = "llm"
            else:
                key = "other"
            if has_grad:
                groups[key]["params"] += 1
                groups[key]["numel"] += param.numel()
        rank0_print(
            "[cycle_debug][grad_stats] "
            f"{tag}: "
            f"params_with_grad={grad_params:,}/{total_params:,} "
            f"numel_with_grad={grad_numel:,}/{total_numel:,} "
            f"groups={groups}"
        )

    def _maybe_init_param_ready_debug(self):
        if not getattr(self.args, "verbose_logging", False):
            return
        if os.getenv("ROSS3D_DEBUG_PARAM_READY") != "1":
            return
        if hasattr(self, "_param_ready_handles"):
            return
        self._param_ready_counts = {}
        self._param_ready_handles = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self._param_ready_counts[name] = 0

            def _make_hook(param_name):
                def _hook(*_):
                    self._param_ready_counts[param_name] += 1
                return _hook

            self._param_ready_handles.append(param.register_hook(_make_hook(name)))
        rank0_print(
            "[checkpoint-debug] ROSS3D_DEBUG_PARAM_READY enabled; "
            f"tracking {len(self._param_ready_counts)} trainable params"
        )

    def _nan_debug_rank0_enabled(self) -> bool:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return False
        return True

    def _audit_model_special_params(self, stage: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        candidates = [model]
        if hasattr(model, "module"):
            candidates.append(model.module)
        for m in candidates:
            targets = [m]
            if hasattr(m, "get_model"):
                try:
                    targets.append(m.get_model())
                except Exception:
                    pass
            for t in targets:
                if t is None:
                    continue
                if hasattr(t, "_audit_special_param_finiteness"):
                    t._audit_special_param_finiteness(stage)
                if hasattr(t, "_audit_special_param_aliases"):
                    t._audit_special_param_aliases(stage)
                if hasattr(t, "_sanitize_mask_token_if_nonfinite"):
                    pre_forward_only = os.getenv("ROSS3D_SANITIZE_MASK_TOKEN_PRE_FORWARD_ONLY", "0") == "1"
                    if (not pre_forward_only) or (stage == "before_first_batch_forward"):
                        t._sanitize_mask_token_if_nonfinite(stage)

    def _format_grad_state(self, grad: Optional[torch.Tensor]):
        if grad is None:
            return "none", None, None, None, None, None
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
        grad_state = "finite" if finite_all else "nonfinite"
        return grad_state, nan_count, inf_count, gmin, gmax, tuple(g.shape)

    def _get_target_debug_params(self):
        model = getattr(self, "model", None)
        if model is None:
            return []
        targets = [
            "mask_token",
            "image_newline",
            "mm_projector.0.weight",
            "mm_projector.0.bias",
            "mm_projector.2.weight",
            "mm_projector.2.bias",
        ]
        return [(name, p) for name, p in model.named_parameters() if any(t in name for t in targets)]

    def _maybe_install_target_grad_hooks(self) -> None:
        if getattr(self, "_nan_debug_target_grad_hooks_installed", False):
            return
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        targets = self._get_target_debug_params()
        self._nan_debug_target_grad_handles = []
        self._nan_debug_grad_hook_counter = 0
        self._nan_debug_first_nonfinite_grad_hook = None

        for name, param in targets:
            def _make_hook(param_name: str):
                def _hook(grad):
                    if not self._nan_debug_rank0_enabled():
                        return grad
                    self._nan_debug_grad_hook_counter += 1
                    grad_state, nan_count, inf_count, gmin, gmax, gshape = self._format_grad_state(grad)
                    micro = getattr(self, "_gradient_accumulation_steps", None)
                    rank0_print(
                        "[NAN_DEBUG][hook] "
                        f"name={param_name} step={int(getattr(self.state, 'global_step', -1))} "
                        f"microstep={micro} order={self._nan_debug_grad_hook_counter} grad_state={grad_state} "
                        f"nan_count={nan_count} inf_count={inf_count} min={gmin} max={gmax} "
                        f"dtype={getattr(grad, 'dtype', None)} shape={gshape}"
                    )
                    if grad_state == "nonfinite" and self._nan_debug_first_nonfinite_grad_hook is None:
                        self._nan_debug_first_nonfinite_grad_hook = {
                            "name": param_name,
                            "step": int(getattr(self.state, 'global_step', -1)),
                            "order": self._nan_debug_grad_hook_counter,
                        }
                        rank0_print(f"[NAN_DEBUG][hook] first_nonfinite_grad_event={self._nan_debug_first_nonfinite_grad_hook}")
                    return grad
                return _hook
            if param.requires_grad:
                self._nan_debug_target_grad_handles.append(param.register_hook(_make_hook(name)))
            else:
                if self._nan_debug_rank0_enabled():
                    rank0_print(f"[NAN_DEBUG][hook_skip] name={name} requires_grad=False")

        self._nan_debug_target_grad_hooks_installed = True

    def _log_target_finiteness_boundary(self, tag: str) -> None:
        if not self._nan_debug_rank0_enabled():
            return
        for name, param in self._get_target_debug_params():
            pdata = param.detach()
            p_finite = torch.isfinite(pdata)
            p_finite_all = bool(p_finite.all().item())
            p_finite_any = bool(p_finite.any().item())
            p_nan = int(torch.isnan(pdata).sum().item())
            p_inf = int(torch.isinf(pdata).sum().item())
            if p_finite_any:
                pvals = pdata[p_finite]
                pmin = float(pvals.min().item())
                pmax = float(pvals.max().item())
            else:
                pmin, pmax = None, None

            grad_state, g_nan, g_inf, gmin, gmax, gshape = self._format_grad_state(param.grad)
            rank0_print(
                f"[NAN_DEBUG][transition] boundary={tag} name={name} "
                f"param_finite_all={p_finite_all} param_nan={p_nan} param_inf={p_inf} param_min={pmin} param_max={pmax} "
                f"grad_state={grad_state} grad_nan={g_nan} grad_inf={g_inf} grad_min={gmin} grad_max={gmax} "
                f"param_dtype={pdata.dtype} param_shape={tuple(pdata.shape)} grad_shape={gshape}"
            )
            first_map = getattr(self, "_nan_debug_first_nonfinite_boundary", None)
            if first_map is None:
                first_map = {}
                setattr(self, "_nan_debug_first_nonfinite_boundary", first_map)
            if name not in first_map and ((not p_finite_all) or (grad_state == "nonfinite")):
                first_map[name] = tag
                rank0_print(f"[NAN_DEBUG][transition] first_nonfinite_boundary name={name} boundary={tag}")

        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "_maybe_log_projector_internal_backward"):
            model._maybe_log_projector_internal_backward(tag)
        if model is not None and hasattr(model, "_maybe_log_newline_packed_grad"):
            model._maybe_log_newline_packed_grad(tag)
        if model is not None and hasattr(model, "_maybe_log_multimodal_backward_chain"):
            model._maybe_log_multimodal_backward_chain(tag)
        if model is not None and hasattr(model, "_nan_debug_validate_layout_snapshot"):
            model._nan_debug_validate_layout_snapshot(tag)
        if model is not None and hasattr(model, "_log_lm_boundary_grad_summary"):
            model._log_lm_boundary_grad_summary(tag)

    def _parse_force_set_to_none(self):
        raw = os.getenv("ROSS3D_FORCE_ZERO_GRAD_SET_TO_NONE", "").strip().lower()
        if raw in {"1", "true", "yes", "y", "none_true"}:
            return True
        if raw in {"0", "false", "no", "n", "none_false"}:
            return False
        return None

    def _maybe_run_inplace_stale_state_audit(self) -> None:
        if os.getenv("ROSS3D_NAN_AUDIT_INPLACE", "0") != "1":
            return
        if bool(getattr(self, "_nan_debug_inplace_audit_done", False)):
            return
        if not self._nan_debug_rank0_enabled():
            return
        mut_patterns = [".data", ".copy_(", ".set_(", ".fill_(", ".zero_(", ".normal_(", ".uniform_(", ".add_(", ".mul_(", ".masked_fill_(", ".scatter_(", ".index_put_("]
        root = Path("ross3d")
        for path in sorted(root.rglob("*.py")):
            try:
                text = path.read_text().splitlines()
            except Exception:
                continue
            f = str(path)
            for i, line in enumerate(text, start=1):
                if any(p in line for p in mut_patterns):
                    rank0_print(f"[NAN_DEBUG][audit] file={f}:{i} line={line.strip()}")
                if ("mask_token" in line or "image_newline" in line) and ("=" in line):
                    rank0_print(f"[NAN_DEBUG][audit][mask_newline_assign] file={f}:{i} line={line.strip()}")
                if "self." in line and "=" in line and "_nan_debug" not in line and "def " not in line:
                    if any(tok in line for tok in ["image", "embed", "feature", "mask", "newline"]):
                        rank0_print(f"[NAN_DEBUG][audit][state_assign] file={f}:{i} line={line.strip()}")
        self._nan_debug_inplace_audit_done = True

    def _manual_one_batch_root_cause_pass(self, model, inputs):
        if not self._nan_debug_rank0_enabled():
            return None
        variant = os.getenv("ROSS3D_MANUAL_ONE_BATCH_VARIANT", "backward_then_second_forward")
        rank0_print(f"[NAN_DEBUG][manual_one_batch] variant={variant}")

        local_inputs = copy.deepcopy(inputs)
        loss, outputs = self._compute_loss_with_global_step(model, local_inputs, return_outputs=True)
        self._log_loss_finiteness(loss, outputs)
        self.accelerator.backward(loss)
        self._log_target_finiteness_boundary("manual_after_backward")

        if variant == "backward_grad_norm_second_forward":
            params = [p for p in model.parameters() if p.grad is not None]
            if len(params) > 0:
                total = torch.zeros((), device=params[0].device)
                for p in params:
                    g = p.grad.detach().float()
                    if torch.isfinite(g).all():
                        total = total + (g * g).sum()
                rank0_print(f"[NAN_DEBUG][manual_one_batch] grad_norm={float(torch.sqrt(total).item()):.6e}")
        elif variant == "backward_grad_clip_second_forward":
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            rank0_print(f"[NAN_DEBUG][manual_one_batch] clip_grad_norm={float(norm):.6e}")
        elif variant == "backward_zero_grad_second_forward":
            opt = getattr(self, "optimizer", None)
            if opt is not None:
                opt.zero_grad(set_to_none=True)

        self._log_target_finiteness_boundary("manual_before_second_forward")
        second_ok = True
        second_err = None
        with torch.no_grad():
            try:
                local_inputs2 = copy.deepcopy(inputs)
                loss2, outputs2 = self._compute_loss_with_global_step(model, local_inputs2, return_outputs=True)
                self._log_loss_finiteness(loss2, outputs2)
            except Exception as e:
                second_ok = False
                second_err = str(e)
        rank0_print(f"[NAN_DEBUG][manual_one_batch] second_forward_ok={second_ok} second_forward_err={second_err}")
        self._log_target_finiteness_boundary("manual_after_second_forward")
        return loss.detach()

    def _maybe_install_zero_grad_debug_hooks(self) -> None:
        if getattr(self, "_nan_debug_zero_grad_hooks_installed", False):
            return
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return

        optimizer = getattr(self, "optimizer", None)
        model = getattr(self, "model", None)

        if optimizer is not None and hasattr(optimizer, "zero_grad"):
            orig_opt_zero_grad = optimizer.zero_grad

            def _opt_zero_grad_wrapper(*args, **kwargs):
                if self._nan_debug_rank0_enabled():
                    opt_type = type(optimizer).__name__
                    path_name = "accelerate-managed zero_grad" if "Accelerated" in opt_type else "optimizer.zero_grad"
                    rank0_print(
                        "[NAN_DEBUG][zero_grad] "
                        f"path={path_name} optimizer_type={opt_type} incoming_kwargs={kwargs}"
                    )
                    self._log_target_finiteness_boundary("before_zero_grad")

                if os.getenv("ROSS3D_SKIP_ZERO_GRAD_ONCE", "0") == "1" and not bool(getattr(self, "_nan_debug_skipped_zero_grad_once", False)):
                    setattr(self, "_nan_debug_skipped_zero_grad_once", True)
                    if self._nan_debug_rank0_enabled():
                        rank0_print("[NAN_DEBUG][zero_grad] skip_zero_grad_once active, skipping this zero_grad call")
                        self._log_target_finiteness_boundary("after_zero_grad_skip")
                    return None

                forced = self._parse_force_set_to_none()
                if forced is not None:
                    kwargs["set_to_none"] = forced

                out = orig_opt_zero_grad(*args, **kwargs)
                if self._nan_debug_rank0_enabled():
                    rank0_print(
                        "[NAN_DEBUG][zero_grad] completed path=optimizer.zero_grad "
                        f"effective_set_to_none={kwargs.get('set_to_none', 'default')}"
                    )
                    self._log_target_finiteness_boundary("after_zero_grad")
                return out

            optimizer.zero_grad = _opt_zero_grad_wrapper

        if optimizer is not None and hasattr(optimizer, "step"):
            orig_opt_step = optimizer.step

            def _opt_step_wrapper(*args, **kwargs):
                rlog("BEFORE_OPTIM")
                out = orig_opt_step(*args, **kwargs)
                rlog("AFTER_OPTIM")
                return out

            optimizer.step = _opt_step_wrapper

        if model is not None and hasattr(model, "zero_grad"):
            orig_model_zero_grad = model.zero_grad

            def _model_zero_grad_wrapper(*args, **kwargs):
                if self._nan_debug_rank0_enabled():
                    rank0_print(
                        "[NAN_DEBUG][zero_grad] path=model.zero_grad "
                        f"model_type={type(model).__name__} incoming_kwargs={kwargs}"
                    )
                    self._log_target_finiteness_boundary("before_model_zero_grad")
                out = orig_model_zero_grad(*args, **kwargs)
                if self._nan_debug_rank0_enabled():
                    self._log_target_finiteness_boundary("after_model_zero_grad")
                return out

            model.zero_grad = _model_zero_grad_wrapper

        scheduler = getattr(self, "lr_scheduler", None)
        if scheduler is not None and hasattr(scheduler, "step"):
            orig_sched_step = scheduler.step

            def _sched_step_wrapper(*args, **kwargs):
                if self._nan_debug_rank0_enabled():
                    opt_executed = bool(getattr(self, "_nan_debug_optimizer_step_executed_last", False))
                    rank0_print(
                        f"[NAN_DEBUG][transition] scheduler_step_start scheduler_type={type(scheduler).__name__} "
                        f"optimizer_step_executed_last={opt_executed}"
                    )
                    self._log_target_finiteness_boundary("before_scheduler_step")
                out = orig_sched_step(*args, **kwargs)
                if self._nan_debug_rank0_enabled():
                    rank0_print("[NAN_DEBUG][transition] scheduler_step_end")
                    self._log_target_finiteness_boundary("after_scheduler_step")
                return out

            scheduler.step = _sched_step_wrapper

        self._nan_debug_zero_grad_hooks_installed = True

    def training_step(self, model, inputs):
        rlog("STEP_START")
        self._maybe_install_backward_markers()
        self._maybe_install_occ_grad_hooks()
        self._reset_occ_hook_fired()
        self._maybe_install_zero_grad_debug_hooks()
        self._maybe_install_target_grad_hooks()
        self._maybe_init_param_ready_debug()
        self._maybe_init_cycle_grad_debug()
        loss = super().training_step(model, inputs)
        self._log_occ_hook_summary()
        self._log_suspect_module_grad_presence("after_backward")
        return loss

    def _capture_target_param_snapshot(self):
        model = getattr(self, "model", None)
        if model is None:
            return {}
        targets = ["mask_token", "image_newline", "mm_projector.0.weight", "mm_projector.0.bias", "mm_projector.2.weight", "mm_projector.2.bias"]
        snap = {}
        for name, param in model.named_parameters():
            if any(t in name for t in targets):
                snap[name] = param.detach().float().clone()
        return snap

    def _log_target_param_mutation(self, before_snapshot, tag: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        for name, param in model.named_parameters():
            if name not in before_snapshot:
                continue
            prev = before_snapshot[name].to(device=param.device)
            cur = param.detach().float()
            if prev.shape != cur.shape:
                rank0_print(f"[NAN_DEBUG][{tag}] param_shape_changed name={name} prev={tuple(prev.shape)} cur={tuple(cur.shape)}")
                continue
            delta = (cur - prev).abs()
            max_delta = float(delta.max().item()) if delta.numel() > 0 else 0.0
            changed = bool(max_delta > 0.0)
            rank0_print(f"[NAN_DEBUG][{tag}] param_mutation name={name} changed={changed} max_abs_delta={max_delta:.6e}")

    def _maybe_freeze_nan_debug_modules(self) -> None:
        combined = os.getenv("ROSS3D_FREEZE_MM_PROJECTOR_DEBUG", "0") == "1"
        freeze_newline_only = os.getenv("ROSS3D_FREEZE_IMAGE_NEWLINE_ONLY", "0") == "1"
        freeze_proj_only = os.getenv("ROSS3D_FREEZE_MM_PROJECTOR_ONLY", "0") == "1"
        if not (combined or freeze_newline_only or freeze_proj_only):
            return
        if bool(getattr(self, "_nan_debug_freeze_applied", False)):
            return
        model = getattr(self, "model", None)
        if model is None:
            return

        frozen = []
        for name, param in model.named_parameters():
            should_freeze = False
            if combined and (("mm_projector" in name) or ("image_newline" in name) or ("mask_token" in name)):
                should_freeze = True
            if freeze_newline_only and ("image_newline" in name):
                should_freeze = True
            if freeze_proj_only and ("mm_projector" in name):
                should_freeze = True
            if should_freeze:
                param.requires_grad = False
                frozen.append(name)

        self._nan_debug_freeze_applied = True
        rank0_print(f"[NAN_DEBUG] froze_params_count={len(frozen)} froze_params_head={frozen[:16]}")

    def _log_target_param_stats(self, named_params, tag: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        targets = [
            "image_newline",
            "mm_projector.0.weight",
            "mm_projector.0.bias",
            "mm_projector.2.weight",
            "mm_projector.2.bias",
            "mask_token",
        ]

        for name, param in named_params:
            if not any(t in name for t in targets):
                continue

            pdata = param.data.detach()
            param_has_nan = bool(torch.isnan(pdata).any().item())
            param_has_inf = bool(torch.isinf(pdata).any().item())
            pfinite = torch.isfinite(pdata)
            if bool(pfinite.any().item()):
                pvals = pdata[pfinite]
                pmin = float(pvals.min().item())
                pmax = float(pvals.max().item())
            else:
                pmin, pmax = None, None

            if param.grad is None:
                grad_has_nan = None
                grad_has_inf = None
                gmin, gmax = None, None
            else:
                gdata = param.grad.detach()
                grad_has_nan = bool(torch.isnan(gdata).any().item())
                grad_has_inf = bool(torch.isinf(gdata).any().item())
                gfinite = torch.isfinite(gdata)
                if bool(gfinite.any().item()):
                    gvals = gdata[gfinite]
                    gmin = float(gvals.min().item())
                    gmax = float(gvals.max().item())
                else:
                    gmin, gmax = None, None

            rank0_print(
                f"[NAN_DEBUG][{tag}][TARGET] "
                f"name={name} "
                f"param_has_nan={param_has_nan} param_has_inf={param_has_inf} "
                f"param_min={pmin} param_max={pmax} "
                f"grad_has_nan={grad_has_nan} grad_has_inf={grad_has_inf} "
                f"grad_min={gmin} grad_max={gmax}"
            )

    def _grad_group_from_name(self, name: str) -> str:
        if "mm_projector" in name:
            return "mm_projector"
        if "image_newline" in name:
            return "image_newline"
        if ("ground_head" in name) or ("predict_box" in name):
            return "ground_head"
        if name.startswith("model.") or name.startswith("lm_head"):
            return "qwen_model"
        return "other"

    def _check_named_params_finiteness(self, named_params, tag: str, max_print: int = 20) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        bad = []
        first_bad = None
        group_keys = ["mm_projector", "image_newline", "qwen_model", "ground_head", "other"]
        group_summary = {key: {"grad": 0, "param": 0} for key in group_keys}

        for name, param in named_params:
            group = self._grad_group_from_name(name)
            if group not in group_summary:
                group = "other"

            if param.grad is not None:
                grad_has_nan = bool(torch.isnan(param.grad).any().item())
                grad_has_inf = bool(torch.isinf(param.grad).any().item())
                if grad_has_nan or grad_has_inf:
                    if first_bad is None:
                        first_bad = (name, "grad", grad_has_nan, grad_has_inf)
                    bad.append((name, "grad", grad_has_nan, grad_has_inf))
                    group_summary[group]["grad"] += 1

            param_has_nan = bool(torch.isnan(param.data).any().item())
            param_has_inf = bool(torch.isinf(param.data).any().item())
            if param_has_nan or param_has_inf:
                if first_bad is None:
                    first_bad = (name, "param", param_has_nan, param_has_inf)
                bad.append((name, "param", param_has_nan, param_has_inf))
                group_summary[group]["param"] += 1

        rank0_print(
            f"[NAN_DEBUG][{tag}] bad_count={len(bad)} first_bad={first_bad} group_summary={group_summary}"
        )
        for row in bad[:max_print]:
            rank0_print(f"[NAN_DEBUG][{tag}] {row}")

    def _check_nonfinite_grads_after_backward_and_guard(self) -> bool:
        model = getattr(self, "model", None)
        if model is None:
            return False

        if not hasattr(self, "_nan_guard_total_hits"):
            self._nan_guard_total_hits = 0
            self._nan_guard_consecutive_hits = 0
            self._nan_guard_max_consecutive_hits = 0
            self._nan_guard_last_hit_step = -1
            self._nan_guard_first_bad_group_hist = {}
            self._nan_guard_first_bad_param_hist = {}

        first_bad = None
        for name, param in model.named_parameters():
            if (not param.requires_grad) or (param.grad is None):
                continue
            if not torch.isfinite(param.grad).all():
                first_bad = name
                break

        if first_bad is None:
            self._nan_guard_consecutive_hits = 0
            self._nan_debug_skip_optimizer_step = False
            self._maybe_log_nan_guard_summary(force=False)
            if self._nan_debug_rank0_enabled():
                rank0_print("[NAN_DEBUG][after_backward] nonfinite_guard_triggered=False")
            return False

        self._nan_guard_total_hits += 1
        self._nan_guard_consecutive_hits += 1
        self._nan_guard_max_consecutive_hits = max(self._nan_guard_max_consecutive_hits, self._nan_guard_consecutive_hits)
        self._nan_guard_last_hit_step = int(self.state.global_step)
        bad_group = self._grad_group_from_name(first_bad)
        self._nan_guard_first_bad_group_hist[bad_group] = int(self._nan_guard_first_bad_group_hist.get(bad_group, 0)) + 1
        self._nan_guard_first_bad_param_hist[first_bad] = int(self._nan_guard_first_bad_param_hist.get(first_bad, 0)) + 1

        if os.getenv("ROSS3D_NAN_DEBUG", "0") == "1":
            if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                rank0_print(
                    "[NAN_DEBUG][after_backward] "
                    f"first_nonfinite_grad_param={first_bad} group={bad_group} "
                    f"global_step={self.state.global_step}"
                )

        self._maybe_log_nan_guard_summary(force=False)

        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        self._nan_debug_skip_optimizer_step = True
        if self._nan_debug_rank0_enabled():
            rank0_print(
                f"[NAN_DEBUG][after_backward] nonfinite_guard_triggered=True first_bad={first_bad} "
                "action=skip_optimizer_step_and_zero_grad"
            )
        return True

    def _maybe_log_nan_guard_summary(self, force: bool = False) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        total_steps = int(getattr(self, "_nan_guard_total_steps", 0))
        if total_steps <= 0:
            return
        interval = int(os.getenv("ROSS3D_NAN_GUARD_SUMMARY_INTERVAL", "50"))
        if (not force) and (total_steps % interval != 0):
            return

        total_hits = int(getattr(self, "_nan_guard_total_hits", 0))
        hit_rate = float(total_hits) / float(total_steps)
        last_hit_step = int(getattr(self, "_nan_guard_last_hit_step", -1))
        steps_since_last_hit = (total_steps - last_hit_step) if last_hit_step >= 0 else -1
        group_hist = dict(sorted(getattr(self, "_nan_guard_first_bad_group_hist", {}).items(), key=lambda kv: kv[1], reverse=True))
        param_hist = sorted(getattr(self, "_nan_guard_first_bad_param_hist", {}).items(), key=lambda kv: kv[1], reverse=True)[:5]
        rank0_print(
            "[NAN_DEBUG][guard_summary] "
            f"steps={total_steps} hits={total_hits} hit_rate={hit_rate:.6f} "
            f"consecutive_current={int(getattr(self, '_nan_guard_consecutive_hits', 0))} "
            f"consecutive_max={int(getattr(self, '_nan_guard_max_consecutive_hits', 0))} "
            f"steps_since_last_hit={steps_since_last_hit} group_hist={group_hist} top_params={param_hist}"
        )
    def _wrap_model(self, model, training=True, dataloader=None):
        model = super()._wrap_model(model, training=training, dataloader=dataloader)
        if os.getenv("ROSS3D_NAN_DEBUG", "0") == "1":
            for m in [model, getattr(model, "module", None)]:
                if m is None:
                    continue
                targets = [m]
                if hasattr(m, "get_model"):
                    try:
                        targets.append(m.get_model())
                    except Exception:
                        pass
                for t in targets:
                    if t is None:
                        continue
                    if hasattr(t, "_audit_special_param_finiteness"):
                        t._audit_special_param_finiteness("after_accelerator_wrapping")
                    if hasattr(t, "_audit_special_param_aliases"):
                        t._audit_special_param_aliases("after_accelerator_wrapping")
        if not training:
            return model
        if not getattr(self.args, "gradient_checkpointing", False):
            return model
        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            if getattr(self.args, "verbose_logging", False):
                rank0_print(
                    "[checkpoint-debug] skipping DDP static graph; "
                    f"fsdp={self.is_fsdp_enabled}, deepspeed={self.is_deepspeed_enabled}"
                )
            return model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) and hasattr(model, "_set_static_graph"):
            rank0_print("[trainer] Enabling DDP static graph to avoid checkpoint re-entrancy issues.")
            model._set_static_graph()
            if getattr(self.args, "verbose_logging", False):
                rank0_print("[checkpoint-debug] DDP static graph enabled")
        if os.getenv("ROSS3D_NAN_DEBUG", "0") == "1":
            for m in [model, getattr(model, "module", None)]:
                if m is None:
                    continue
                targets = [m]
                if hasattr(m, "get_model"):
                    try:
                        targets.append(m.get_model())
                    except Exception:
                        pass
                for t in targets:
                    if t is None:
                        continue
                    if hasattr(t, "_audit_special_param_finiteness"):
                        t._audit_special_param_finiteness("after_ddp_wrapping")
                    if hasattr(t, "_audit_special_param_aliases"):
                        t._audit_special_param_aliases("after_ddp_wrapping")
        return model

    def _log_optimizer_state(self, tag: str) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        if not getattr(getattr(model, "config", None), "cycle_debug_memory", False):
            return
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        grad_numel = 0
        grad_bytes = 0
        param_numel = 0
        param_bytes = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            param_numel += param.numel()
            param_bytes += param.numel() * param.element_size()
            if param.grad is not None:
                grad_numel += param.grad.numel()
                grad_bytes += param.grad.numel() * param.grad.element_size()
        state_numel = 0
        state_bytes = 0
        state_tensors = 0
        for state in optimizer.state.values():
            if isinstance(state, dict):
                values = state.values()
            else:
                values = [state]
            for value in values:
                if torch.is_tensor(value):
                    state_tensors += 1
                    state_numel += value.numel()
                    state_bytes += value.numel() * value.element_size()
        rank0_print(
            "[trainer][optimizer_state] "
            f"{tag}: params={param_numel:,} "
            f"({param_bytes / 1024 ** 3:.2f}GB), "
            f"grads={grad_numel:,} "
            f"({grad_bytes / 1024 ** 3:.2f}GB), "
            f"state_tensors={state_tensors:,}, "
            f"state_numel={state_numel:,} "
            f"({state_bytes / 1024 ** 3:.2f}GB)"
        )

    def _log_optimizer_param_counts(self, tag: str) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        if not getattr(getattr(model, "config", None), "cycle_debug_optimizer", False):
            return
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        total_numel = 0
        unique_params = set()
        group_counts = []
        for group in optimizer.param_groups:
            group_numel = 0
            for param in group["params"]:
                group_numel += param.numel()
                unique_params.add(param)
            group_counts.append(group_numel)
            total_numel += group_numel
        unique_numel = sum(param.numel() for param in unique_params)
        rank0_print(
            "[trainer][optimizer_param_counts] "
            f"{tag}: total_numel={total_numel:,}, "
            f"unique_numel={unique_numel:,}, "
            f"group_numel={group_counts}"
        )

    def _log_cuda_memory(self, tag: str) -> None:
        if not torch.cuda.is_available():
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        if not getattr(getattr(model, "config", None), "cycle_debug_memory", False):
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        rank0_print(
            "[trainer][cuda_mem] "
            f"{tag}: allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, "
            f"max_allocated={max_alloc:.2f}GB"
        )

    def optimizer_step(self, *args, **kwargs):
        model = getattr(self, "model", None)
        if model is not None:
            self._log_target_finiteness_boundary("before_optimizer_step")
            self._check_named_params_finiteness(model.named_parameters(), "before_optimizer_step")
            self._log_target_param_stats(model.named_parameters(), "before_optimizer_step")
        self._log_nonfinite_grad_param_debug("before_optimizer_step")
        self._log_cuda_memory("before_optimizer_step")
        self._log_optimizer_state("before_optimizer_step")
        self._log_optimizer_param_counts("before_optimizer_step")

        ddp_model = model
        if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
            rank0_print(
                "[NAN_DEBUG][ddp] "
                f"find_unused_parameters={getattr(ddp_model, 'find_unused_parameters', 'unknown')} "
                f"static_graph={getattr(ddp_model, 'static_graph', 'unknown')}"
            )

        accelerator = getattr(self, "accelerator", None)
        scaler = getattr(accelerator, "scaler", None) if accelerator is not None else None
        rank0_print(f"[NAN_DEBUG][amp] autocast_enabled_before_opt_step={torch.is_autocast_enabled()}")
        if scaler is not None:
            rank0_print(f"[NAN_DEBUG][amp] scaler_present=True scale_before={float(scaler.get_scale()):.6e}")
        else:
            rank0_print("[NAN_DEBUG][amp] scaler_present=False")

        if model is not None and os.getenv("ROSS3D_DEBUG_CLIP_GRAD", "0") == "1":
            rank0_print("[NAN_DEBUG][transition] grad_clip_start")
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            rank0_print(f"[NAN_DEBUG][transition] grad_clip_end total_norm={float(total_norm):.6e}")

        guard_skip = bool(getattr(self, "_nan_debug_skip_optimizer_step", False))
        env_skip = os.getenv("ROSS3D_SKIP_OPT_STEP", "0") == "1"
        if env_skip or guard_skip:
            reason = "env" if env_skip else "nonfinite_guard"
            rank0_print(f"[NAN_DEBUG][transition] optimizer_step_skipped=True reason={reason}")
            result = None
            self._nan_debug_optimizer_step_executed_last = False
            self._nan_debug_skip_optimizer_step = False
        else:
            rank0_print("[NAN_DEBUG][transition] optimizer_step_skipped=False executing=True")
            result = super().optimizer_step(*args, **kwargs)
            self._nan_debug_optimizer_step_executed_last = True

        rank0_print(f"[NAN_DEBUG][amp] autocast_enabled_after_opt_step={torch.is_autocast_enabled()}")
        if scaler is not None:
            rank0_print(f"[NAN_DEBUG][amp] scale_after={float(scaler.get_scale()):.6e}")
        if model is not None:
            self._log_target_finiteness_boundary("after_optimizer_step")
            self._check_named_params_finiteness(model.named_parameters(), "after_optimizer_step")
            self._log_target_param_stats(model.named_parameters(), "after_optimizer_step")
        self._log_nonfinite_grad_param_debug("after_optimizer_step")
        self._log_cuda_memory("after_optimizer_step")
        self._log_optimizer_state("after_optimizer_step")
        self._log_optimizer_param_counts("after_optimizer_step")
        return result

    def _log_nonfinite_grad_param_debug(self, tag: str) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        count = int(getattr(self, "_nan_debug_opt_count", 0))
        max_logs = int(os.getenv("ROSS3D_NAN_DEBUG_MAX", "64"))
        if count >= max_logs:
            return
        setattr(self, "_nan_debug_opt_count", count + 1)

        model = getattr(self, "model", None)
        if model is None:
            return

        groups = {
            "vision_tower": [],
            "vision_resampler": [],
            "mm_projector": [],
            "image_newline": [],
            "mask_token": [],
            "occupancy": [],
            "other": [],
        }

        def _group_name(param_name: str) -> str:
            if "vision_tower" in param_name:
                return "vision_tower"
            if "vision_resampler" in param_name:
                return "vision_resampler"
            if "mm_projector" in param_name:
                return "mm_projector"
            if "image_newline" in param_name:
                return "image_newline"
            if "mask_token" in param_name:
                return "mask_token"
            if ("occupancy" in param_name) or ("occ_" in param_name):
                return "occupancy"
            return "other"

        for name, param in model.named_parameters():
            groups[_group_name(name)].append((name, param))

        for gname, params in groups.items():
            if len(params) == 0:
                continue

            grad_nonfinite = False
            first_grad_nonfinite_name = None
            grad_norm_sq = torch.zeros((), device=params[0][1].device)
            grad_norm_is_finite = True
            for name, param in params:
                if param.grad is None:
                    continue
                g = param.grad.detach()
                if not torch.isfinite(g).all():
                    grad_nonfinite = True
                    if first_grad_nonfinite_name is None:
                        first_grad_nonfinite_name = name
                if grad_norm_is_finite:
                    if torch.isfinite(g).all():
                        grad_norm_sq = grad_norm_sq + (g.float() * g.float()).sum()
                    else:
                        grad_norm_is_finite = False

            param_nonfinite = False
            first_param_nonfinite_name = None
            for name, param in params:
                p = param.detach()
                if not torch.isfinite(p).all():
                    param_nonfinite = True
                    first_param_nonfinite_name = name
                    break

            grad_norm_msg = "nan"
            if grad_norm_is_finite:
                grad_norm_msg = f"{float(torch.sqrt(grad_norm_sq).item()):.6e}"

            rank0_print(
                "[NAN_DEBUG][optimizer] "
                f"tag={tag} group={gname} grad_nonfinite={grad_nonfinite} "
                f"first_grad_nonfinite={first_grad_nonfinite_name} grad_norm={grad_norm_msg} "
                f"param_nonfinite={param_nonfinite} first_param_nonfinite={first_param_nonfinite_name}"
            )

    def _log_loss_finiteness(self, loss, outputs) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        count = int(getattr(self, "_nan_debug_loss_count", 0))
        max_logs = int(os.getenv("ROSS3D_NAN_DEBUG_MAX", "64"))
        if count >= max_logs:
            return
        setattr(self, "_nan_debug_loss_count", count + 1)

        def _fmt_tensor_finite(name: str, value):
            if value is None:
                return f"{name}=None"
            if not torch.is_tensor(value):
                return f"{name}=non_tensor"
            v = value.detach()
            finite = bool(torch.isfinite(v).all().item())
            msg = f"{name}=finite:{finite}"
            if v.numel() == 1:
                msg += f" val={float(v.float().item()):.6e}"
            else:
                msg += f" shape={tuple(v.shape)}"
            return msg

        parts = [_fmt_tensor_finite("loss", loss)]
        for key in ["lm_loss", "vm_loss", "bev_loss", "cycle_loss", "occ_geom_loss", "occ_temp_loss"]:
            parts.append(_fmt_tensor_finite(key, outputs.get(key, None) if isinstance(outputs, dict) else None))
        rank0_print("[NAN_DEBUG][loss] " + " | ".join(parts))

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")

        # create accelerator object
        accelerator_init_kwargs = {
            "split_batches": self.args.split_batches,
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "gradient_accumulation_plugin": gradient_accumulation_plugin,
            "kwargs_handlers": [accelerator_kwargs],
        }
        if "dispatch_batches" in inspect.signature(Accelerator).parameters:
            accelerator_init_kwargs["dispatch_batches"] = self.args.dispatch_batches
        self.accelerator = Accelerator(**accelerator_init_kwargs)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_task_length:
            lengths = self.train_dataset.task_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_task=True
            )
        elif self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            masking_enabled = (
                float(getattr(getattr(opt_model, "config", None), "view_mask_ratio", 0.0)) > 0.0
                or float(getattr(getattr(opt_model, "config", None), "view_mask_prob", 0.0)) > 0.0
            )
            skip_mask_token = not masking_enabled

            def _use_param(name, param):
                if not param.requires_grad:
                    return False
                if skip_mask_token and ("mask_token" in name):
                    return False
                return True

            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if self.args.mm_inv_projector_lr is not None:
                lr_mapper["mm_inv_projector"] = self.args.mm_inv_projector_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and _use_param(n, p))],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and _use_param(n, p))],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and _use_param(n, p))],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and _use_param(n, p))],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and _use_param(n, p))],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and _use_param(n, p))],
                        "weight_decay": 0.0,
                    },
                ]

            if self.args.verbose_logging:
                inv_params = [(n, p) for n, p in opt_model.named_parameters() if "mm_inv_projector" in n and p.requires_grad]
                inv_param_set = {p for _, p in inv_params}
                grouped_inv_param_set = {
                    p for group in optimizer_grouped_parameters for p in group["params"] if p in inv_param_set
                }
                inv_total_numel = sum(p.numel() for p in inv_param_set)
                inv_grouped_numel = sum(p.numel() for p in grouped_inv_param_set)
                rank0_print(
                    "[mm_inv_projector][debug] "
                    f"optimizer_params={inv_grouped_numel:,}/{inv_total_numel:,}"
                )
                if inv_param_set and inv_grouped_numel == 0:
                    rank0_print(
                        "[mm_inv_projector][debug] WARNING: trainable params missing from optimizer groups."
                    )

            if os.getenv("ROSS3D_NAN_DEBUG", "0") == "1":
                rank0_print(
                    "[NAN_DEBUG][optimizer_groups] "
                    f"masking_enabled={masking_enabled} skip_mask_token={skip_mask_token}"
                )
                param_ptr_seen = set()
                duplicated = 0
                for group in optimizer_grouped_parameters:
                    for p in group["params"]:
                        ptr = p.data_ptr()
                        if ptr in param_ptr_seen:
                            duplicated += 1
                        else:
                            param_ptr_seen.add(ptr)
                rank0_print(
                    "[NAN_DEBUG][optimizer_groups] "
                    f"group_count={len(optimizer_grouped_parameters)} duplicated_param_tensors={duplicated}"
                )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if (
                optimizer_cls.__name__ == "AdamW"
                and hasattr(self.args, "adamw_use_foreach")
                and self.args.adamw_use_foreach is not None
            ):
                optimizer_kwargs["foreach"] = self.args.adamw_use_foreach

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

            rank0_print(self.optimizer)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(Ross3DTrainer, self)._save_checkpoint(model, trial, metrics)
    
    def save_my_lora_ckpt(self, output_dir, args, model):
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if args.local_rank == 0 or args.local_rank == -1:
            model.config.save_pretrained(output_dir)
            model.save_pretrained(output_dir, state_dict=state_dict)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(Ross3DTrainer, self)._save(output_dir, state_dict)

    def _log_loss_gradient_attribution(self, outputs) -> None:
        if os.getenv("ROSS3D_NAN_DEBUG_LOSS_ATTRIB", "0") != "1":
            return
        if not self._nan_debug_rank0_enabled():
            return
        if not isinstance(outputs, dict):
            return
        targets = self._get_target_debug_params()
        if len(targets) == 0:
            return

        loss_keys = ["lm_loss", "occ_temp_loss", "occ_geom_loss", "vm_loss", "bev_loss", "cycle_loss", "loss"]
        for key in loss_keys:
            loss_term = outputs.get(key, None)
            if loss_term is None or (not torch.is_tensor(loss_term)):
                continue
            grads = torch.autograd.grad(loss_term, [p for _, p in targets], retain_graph=True, allow_unused=True)
            for (name, _), g in zip(targets, grads):
                grad_state, g_nan, g_inf, gmin, gmax, gshape = self._format_grad_state(g)
                rank0_print(
                    "[NAN_DEBUG][loss_attrib] "
                    f"loss={key} target={name} grad_state={grad_state} "
                    f"nan_count={g_nan} inf_count={g_inf} min={gmin} max={gmax} shape={gshape}"
                )

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss, outputs = self._compute_loss_with_global_step(model, inputs, return_outputs=True)
        self._log_loss_finiteness(loss, outputs)
        self._log_loss_gradient_attribution(outputs)

        log_dict = {}
        self._cycle_loss_active = outputs.get("cycle_loss", None) is not None

        if outputs.get('lm_loss', None) is not None:
            log_dict["lm_loss"] = round(outputs['lm_loss'].item(), 4)

        if outputs.get('vm_loss', None) is not None:
            vm_loss = outputs['vm_loss']
            log_dict["vm_loss"] = round(vm_loss.item(), 4)

            if outputs.get('bev_loss', None) is not None:
                bev_loss = outputs['bev_loss']
                log_dict["bev_loss"] = round(bev_loss.item(), 4)


        # Hanwliu
        if "cycle_loss" in outputs:
            log_dict["cycle_loss"] = round(outputs["cycle_loss"].item(), 4)
        if outputs.get("occ_geom_loss", None) is not None:
            log_dict["occ_geom_loss"] = round(outputs["occ_geom_loss"].item(), 4)
            occ_geom_component_keys = [
                "occ_geom_mask_loss",
                "occ_geom_box_loss",
                "occ_geom_ctr_loss",
                "occ_geom_vis_loss",
                "occ_geom_mask_bce_loss",
                "occ_geom_mask_dice_loss",
                "occ_geom_box_l1_loss",
                "occ_geom_box_giou_loss",
            ]
            for key in occ_geom_component_keys:
                value = outputs.get(key, None)
                if value is not None:
                    log_dict[key] = round(value.item(), 4)
        if outputs.get("occ_temp_loss", None) is not None:
            log_dict["occ_temp_loss"] = round(outputs["occ_temp_loss"].item(), 4)

        if len(log_dict) > 0:
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def _compute_loss_with_global_step(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs["global_step"] = self.state.global_step
        rlog("BEFORE_FORWARD")
        outputs = model(**inputs)
        rlog("AFTER_FORWARD")
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        rlog("BEFORE_LOSS")
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        rlog("AFTER_LOSS")
        return (loss, outputs) if return_outputs else loss
