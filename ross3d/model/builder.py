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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from ross3d.model import *
from ross3d.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ross3d.utils import rank0_print
from types import SimpleNamespace


def _is_llava_or_multimodal(model_path, model_name, customized_config=None):
    cfg = customized_config
    if cfg is None:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            warnings.warn(f"Failed to infer model type from config at {model_path}: {e}")
            return False

    model_type = getattr(cfg, "model_type", None)
    architectures = getattr(cfg, "architectures", None) or []

    if model_type == "llava":
        return True

    return any("ross3d" in arch.lower() for arch in architectures)


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16",attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        import pdb;pdb.set_trace()

    if customized_config is not None:
        kwargs["config"] = customized_config

    detected_multimodal = _is_llava_or_multimodal(model_path, model_name, customized_config=customized_config)

    if "multimodal" in kwargs:
        is_multimodal = bool(kwargs.pop("multimodal"))
    else:
        is_multimodal = detected_multimodal

    if is_multimodal:
        # Load multimodal model based on config architecture, not folder name
        base_cfg = kwargs.get("config")
        if base_cfg is None:
            base_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        architectures = getattr(base_cfg, "architectures", None) or []

        if any("Ross3DQwenForCausalLM" in arch for arch in architectures):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            from ross3d.model.language_model.ross3d_qwen import Ross3DQwenConfig

            if overwrite_config is not None:
                ross_cfg = Ross3DQwenConfig.from_pretrained(model_path)
                rank0_print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(ross_cfg, k, v)
                model = Ross3DQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=ross_cfg, trust_remote_code=True, **kwargs)
            else:
                model = Ross3DQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, trust_remote_code=True, **kwargs)
        else:
            raise ValueError(f"Multimodal model architecture not supported: {architectures}")

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # Hanwliu
    if hasattr(model.config, "text_config") and isinstance(model.config.text_config, dict):
        model.config.text_config = SimpleNamespace(**model.config.text_config)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None

    if is_multimodal:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
