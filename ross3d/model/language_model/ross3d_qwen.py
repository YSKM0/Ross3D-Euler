#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import os
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ross3d.model.ross3d_arch import Ross3DMetaModel, Ross3DMetaForCausalLM, CausalLMOutputWithPastRoss
from transformers import Qwen2Config
from .qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from ross3d.utils import rank_print


class Ross3DQwenConfig(Qwen2Config):
    model_type = "ross3d_qwen"

    # Hanwliu
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cycle_consist = kwargs.get("cycle_consist", False)
        self.cycle_consist_v2 = kwargs.get("cycle_consist_v2", False)
        self.cycle_consist_weight = kwargs.get("cycle_consist_weight", 1.0)
        self.cycle_num_walks = kwargs.get("cycle_num_walks", None)
        self.cycle_geo_temp = kwargs.get("cycle_geo_temp", 0.10)
        self.cycle_geo_sigma = kwargs.get("cycle_geo_sigma", None)
        self.cycle_topk = kwargs.get("cycle_topk", 32)
        self.cycle_filter_positive_depth = kwargs.get("cycle_filter_positive_depth", True)
        self.cycle_debug_memory = kwargs.get("cycle_debug_memory", False)
        self.cycle_debug_grad = kwargs.get("cycle_debug_grad", False)
        self.cycle_debug_optimizer = kwargs.get("cycle_debug_optimizer", False)
        self.cycle_detach_hidden_states = kwargs.get("cycle_detach_hidden_states", True)
        self.use_3d_coordinate = kwargs.get("use_3d_coordinate", True)
        self.occupancy_projector_dim = kwargs.get("occupancy_projector_dim", None)
        self.enable_occ_geom_loss = kwargs.get("enable_occ_geom_loss", False)
        self.occ_geom_loss_weight = kwargs.get("occ_geom_loss_weight", 0.0)
        self.occ_geom_mask_weight = kwargs.get("occ_geom_mask_weight", 1.0)
        self.occ_geom_box_weight = kwargs.get("occ_geom_box_weight", 1.0)
        self.occ_geom_ctr_weight = kwargs.get("occ_geom_ctr_weight", 1.0)
        self.occ_geom_vis_weight = kwargs.get("occ_geom_vis_weight", 1.0)
        self.occ_geom_mask_dice_weight = kwargs.get("occ_geom_mask_dice_weight", 0.5)
        self.occ_geom_box_giou_weight = kwargs.get("occ_geom_box_giou_weight", 1.0)
        self.occ_geom_center_alpha = kwargs.get("occ_geom_center_alpha", 0.1)
        self.occ_geom_eps = kwargs.get("occ_geom_eps", 1e-6)
        self.enable_occ_temp_loss = kwargs.get("enable_occ_temp_loss", False)
        self.occ_temp_loss_weight = kwargs.get("occ_temp_loss_weight", 0.0)
        self.occ_temp_eps = kwargs.get("occ_temp_eps", 1e-6)
        self.occ_temp_min_frames = kwargs.get("occ_temp_min_frames", 2)
        self.occ_temp_pos_weight = kwargs.get("occ_temp_pos_weight", 1.0)
        self.occ_temp_same_min_margin = kwargs.get("occ_temp_same_min_margin", 0.10)
        self.occ_temp_same_max_margin = kwargs.get("occ_temp_same_max_margin", 0.25)
        self.occ_temp_same_weight = kwargs.get("occ_temp_same_weight", 1.0)
        self.occ_temp_diff_margin = kwargs.get("occ_temp_diff_margin", 0.30)
        self.occ_temp_diff_weight = kwargs.get("occ_temp_diff_weight", 1.0)


class Ross3DQwenModel(Ross3DMetaModel, Qwen2Model):
    config_class = Ross3DQwenConfig

    def __init__(self, config: Qwen2Config):
        super(Ross3DQwenModel, self).__init__(config)


class Ross3DQwenForCausalLM(Qwen2ForCausalLM, Ross3DMetaForCausalLM):
    config_class = Ross3DQwenConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        # Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "ross3d_qwen"
        config.rope_scaling = None

        # Hanwliu temperature
        init_tau = getattr(config, "cycle_temp", 0.07)
        self.temperature_app = nn.Parameter(torch.tensor(init_tau))

        self.model = Ross3DQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if hasattr(config, "ground_head_type") and config.ground_head_type is not None:
            self.ground_head_type = config.ground_head_type
            if config.ground_head_type == "mlp":
                # self.ground_head = nn.Sequential(
                #     nn.Linear(config.hidden_size, config.ground_head_hidden_size),
                #     nn.ReLU(),
                #     nn.LayerNorm(config.ground_head_hidden_size),
                #     nn.Linear(config.ground_head_hidden_size, 6)
                # )
                self.ground_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    # nn.GELU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            elif config.ground_head_type == "score":
                self.ground_head_temperature = config.ground_head_temperature
                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    # nn.GELU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    # nn.GELU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_score = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    # nn.GELU(),
                    nn.Linear(1024, 1),
                )
            elif config.ground_head_type == "infonce":
                # self.ground_head_temperature = nn.Parameter(torch.tensor(config.ground_head_temperature))
                try:
                    self.ground_head_temperature = config.ground_head_temperature
                except:
                    self.ground_head_temperature = 0.07
                self.ground_head_zero_target = torch.nn.Parameter(torch.randn(config.hidden_size))

                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    # nn.GELU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    # nn.GELU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            else:
                raise NotImplementedError
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position: Optional[torch.FloatTensor] = None,
        video_dict = None,
        use_object_proposals: bool = False,
        box_labels = None,
        global_step: Optional[int] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            if not self.training or getattr(self.config, "view_mask_prob", 0) == 0:
                replace_with_mask_token = False
            else:
                replace_with_mask_token = (not use_object_proposals) and (
                            global_step % int(1 / self.config.view_mask_prob) == 0)
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                object_features,
                object_boxes,
                boi_ids,
                eoi_ids,
                newline_ids,
                mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
                video_dict,
                use_object_proposals=use_object_proposals,
                replace_with_mask_token=replace_with_mask_token,
            )
        else:
            boi_ids, eoi_ids, newline_ids, mask = None, None, None, None

        if use_object_proposals:
            return self.predict_box(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                object_features=object_features,
                object_boxes=object_boxes,
                box_labels=box_labels,
                boi_ids=boi_ids,
                eoi_ids=eoi_ids,
                images=images,
                newline_ids=newline_ids,
                mask=mask,
                global_step=global_step,
                video_dict=video_dict,
            )


        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return self.inner_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                boi_ids=boi_ids,
                eoi_ids=eoi_ids,
                images=images,
                newline_ids=newline_ids,
                mask=mask,
                global_step=global_step,
                video_dict=video_dict,
            )


    def _dtype_debug_should_log(self) -> bool:
        if os.getenv("ROSS3D_DTYPE_DEBUG", "0") != "1":
            return False
        max_logs = int(os.getenv("ROSS3D_DTYPE_DEBUG_MAX", "2"))
        count = getattr(self, "_dtype_debug_log_count", 0)
        if count >= max_logs:
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return False
        setattr(self, "_dtype_debug_log_count", count + 1)
        return True

    def _nan_debug_should_log(self) -> bool:
        if os.getenv("ROSS3D_NAN_DEBUG", "0") != "1":
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return False
        max_logs = int(os.getenv("ROSS3D_NAN_DEBUG_MAX", "64"))
        count = int(getattr(self, "_nan_debug_log_count", 0))
        if count >= max_logs:
            return False
        setattr(self, "_nan_debug_log_count", count + 1)
        return True

    def _nan_debug_tensor_stats(self, tag: str, t: Optional[torch.Tensor]) -> None:
        if (t is None) or (not torch.is_tensor(t)):
            return
        if not self._nan_debug_should_log():
            return
        x = t.detach()
        finite = torch.isfinite(x)
        finite_any = bool(finite.any().item())
        nan_count = int(torch.isnan(x).sum().item())
        inf_count = int(torch.isinf(x).sum().item())
        if finite_any:
            vals = x[finite]
            min_val = float(vals.min().item())
            max_val = float(vals.max().item())
            minmax_msg = f"min={min_val:.6e} max={max_val:.6e}"
        else:
            minmax_msg = "min=NA max=NA"
        rank_print(
            "[NAN_DEBUG][predict_box] "
            f"tag={tag} shape={tuple(x.shape)} dtype={x.dtype} "
            f"finite_all={bool(finite.all().item())} nan_count={nan_count} inf_count={inf_count} {minmax_msg}"
        )

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boi_ids: Optional[List[int]] = None,
        eoi_ids: Optional[List[int]] = None,
        images: Optional[torch.Tensor] = None,
        newline_ids: Optional[torch.Tensor] = None,
        mask: Optional[List[torch.Tensor]] = None,
        global_step: Optional[int] = 0,
        video_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if self._dtype_debug_should_log():
            rank_print(
                "[DTYPE_DEBUG][inner_forward][before self.model] "
                f"inputs_embeds={getattr(inputs_embeds, 'dtype', None)} "
                f"attention_mask={getattr(attention_mask, 'dtype', None)} "
                f"position_ids={getattr(position_ids, 'dtype', None)} "
                f"labels={getattr(labels, 'dtype', None)} "
                f"occ_enabled={(bool(getattr(self.config, 'enable_occ_geom_loss', False)) or bool(getattr(self.config, 'enable_occ_temp_loss', False)))}"
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self._dtype_debug_should_log():
            rank_print(
                "[DTYPE_DEBUG][inner_forward][after self.model] "
                f"hidden_states={hidden_states.dtype} logits_weight={self.lm_head.weight.dtype}"
            )
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = hidden_states.new_zeros(())
        lm_loss = hidden_states.new_zeros(())
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            lm_loss = loss.detach().clone()

        vm_loss, bev_loss = None, None
        vm_enabled = getattr(self.config, "enable_vm_loss", True)
        bev_enabled = getattr(self.config, "enable_bev_loss", True)
        if self.training and getattr(self.config, 'ross_enable', False) and vm_enabled and (
            getattr(self.config, "view_mask_prob", 0.0) > 0.0
            or getattr(self.config, "view_mask_ratio", 0.0) > 0.0
        ):
            # vm_loss = self.compute_vm_loss_v2(images, hidden_states, boi_ids, eoi_ids, newline_ids, mask)
            vm_loss = self.compute_vm_loss(images, hidden_states, boi_ids, eoi_ids, newline_ids, mask)
            loss = loss + vm_loss
            if getattr(self.config, 'ross_multi_task', False) and bev_enabled:
                bev_loss = self.compute_vm_loss_bev(images, hidden_states, boi_ids, eoi_ids, newline_ids,
                                                    video_dict["bev_image"], mask)
                loss = loss + bev_loss

        occupancy_aux_outputs = None
        occ_geom_enabled = bool(getattr(self.config, "enable_occ_geom_loss", False))
        occ_temp_enabled = bool(getattr(self.config, "enable_occ_temp_loss", False))
        occ_aux_enabled = occ_geom_enabled or occ_temp_enabled
        if self.training and occ_aux_enabled and (video_dict is not None):
            occupancy_aux_outputs = self.extract_occupancy_object_embeddings(
                hidden_states=hidden_states,
                boi_ids=boi_ids,
                eoi_ids=eoi_ids,
                newline_ids=newline_ids,
                video_dict=video_dict,
            )

        occ_geom_loss = None
        if (
            self.training
            and occ_geom_enabled
            and (occupancy_aux_outputs is not None)
            and (video_dict is not None)
        ):
            occ_geom_loss = self.compute_occupancy_geometry_loss(
                occupancy_aux_outputs=occupancy_aux_outputs,
                video_dict=video_dict,
            ).float()
            loss = loss + getattr(self.config, "occ_geom_loss_weight", 0.0) * occ_geom_loss

        occ_temp_loss = None
        if (
            self.training
            and occ_temp_enabled
            and (occupancy_aux_outputs is not None)
        ):
            occ_temp_loss = self.compute_occupancy_temporal_loss(
                occupancy_aux_outputs=occupancy_aux_outputs,
            ).float()
            loss = loss + getattr(self.config, "occ_temp_loss_weight", 0.0) * occ_temp_loss

        # Hanwliu
        cycle_loss = None
        if self.training and (
            getattr(self.config, "cycle_consist_v2", False)
            or getattr(self.config, "cycle_consist", False)
        ):
            cycle_hidden_states = hidden_states
            if getattr(self.config, "cycle_detach_hidden_states", False):
                cycle_hidden_states = hidden_states.detach()
                if getattr(self.config, "verbose_logging", False):
                    rank_print(
                        "[cycle_consistency_loss] using detached hidden_states; "
                        f"hidden_states_requires_grad={hidden_states.requires_grad}, "
                        f"detached_requires_grad={cycle_hidden_states.requires_grad}"
                    )
            cycle_kwargs = dict(
                hidden_states=cycle_hidden_states,
                boi_ids=boi_ids,
                eoi_ids=eoi_ids,
                newline_ids=newline_ids,
                video_dict=video_dict,  # <-- IMPORTANT
                mask=None,
                num_walks=getattr(self.config, "cycle_num_walks", None),
                temperature_app=self.temperature_app,
                temperature_geo=getattr(self.config, "cycle_geo_temp", 0.10),
                geo_sigma=getattr(self.config, "cycle_geo_sigma", None),  # auto if None
                topk=getattr(self.config, "cycle_topk", 32),
            )
            if getattr(self.config, "cycle_consist_v2", False):
                cycle_loss = self.compute_cycle_consistency_loss_v2(**cycle_kwargs)
            else:
                cycle_loss = self.compute_cycle_consistency_loss(**cycle_kwargs)
            loss = loss + getattr(self.config, "cycle_consist_weight", 1.0) * cycle_loss


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastRoss(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            vm_loss=vm_loss,
            bev_loss=bev_loss,
            cycle_loss=cycle_loss, # Hanwliu
            occupancy_aux_outputs=occupancy_aux_outputs,
            occ_geom_loss=occ_geom_loss,
            occ_temp_loss=occ_temp_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                object_features,
                object_boxes,
                boi_ids,
                eoi_ids,
                newline_ids,
                mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                video_dict=kwargs.get("video_dict", None),
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    
    @torch._dynamo.disable
    def _run_lm_eager(self, **kwargs):
        return self.model(**kwargs)

    def _should_log_rank0(self) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return False
        return True

    def _get_lm_attn_impl(self) -> Optional[str]:
        for cfg in [getattr(self, "config", None), getattr(self.model, "config", None)]:
            if cfg is None:
                continue
            impl = getattr(cfg, "_attn_implementation", None)
            if impl is not None:
                return impl
        return None

    def _set_lm_attn_impl(self, impl: str) -> None:
        for cfg in [getattr(self, "config", None), getattr(self.model, "config", None)]:
            if cfg is None:
                continue
            setattr(cfg, "_attn_implementation", impl)

    def _lm_boundary_grad_debug_enabled(self) -> bool:
        return os.getenv("ROSS3D_LM_BOUNDARY_GRAD_DEBUG", "0") == "1"

    def _log_tensor_grad_stats(self, tag: str, tensor: Optional[torch.Tensor]) -> None:
        if not self._lm_boundary_grad_debug_enabled():
            return
        if not self._should_log_rank0():
            return
        if tensor is None or (not torch.is_tensor(tensor)):
            rank_print(f"[NAN_DEBUG][lm_boundary] tag={tag} grad_state=none")
            return
        grad = tensor.grad
        if grad is None:
            rank_print(f"[NAN_DEBUG][lm_boundary] tag={tag} grad_state=none")
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
        rank_print(
            f"[NAN_DEBUG][lm_boundary] tag={tag} grad_finite_all={finite_all} "
            f"nan_count={nan_count} inf_count={inf_count} min={gmin} max={gmax} "
            f"dtype={g.dtype} shape={tuple(g.shape)}"
        )

    def _retain_lm_boundary_grad(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        if not self._lm_boundary_grad_debug_enabled():
            return
        if tensor is None or (not torch.is_tensor(tensor)) or (not tensor.requires_grad):
            return
        tensor.retain_grad()
        store = getattr(self, "_nan_debug_lm_boundary_tensors", None)
        if store is None:
            store = {}
            setattr(self, "_nan_debug_lm_boundary_tensors", store)
        store[name] = tensor

    def _log_lm_boundary_grad_summary(self, boundary_tag: str) -> None:
        if not self._lm_boundary_grad_debug_enabled():
            return
        store = getattr(self, "_nan_debug_lm_boundary_tensors", {})
        for key in ["lm_inputs_embeds", "lm_hidden_states", "ground_hidden", "lm_scores"]:
            self._log_tensor_grad_stats(f"{boundary_tag}/{key}", store.get(key, None))

    def predict_box(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
        video_dict=None,
        object_features=None,
        object_boxes=None,
        box_labels=None,
        boi_ids: Optional[List[int]] = None,
        eoi_ids: Optional[List[int]] = None,
        newline_ids: Optional[torch.Tensor] = None,
        mask: Optional[List[torch.Tensor]] = None,
        global_step: Optional[int] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if self._dtype_debug_should_log():
            rank_print(
                "[DTYPE_DEBUG][predict_box][before self.model] "
                f"inputs_embeds={getattr(inputs_embeds, 'dtype', None)} "
                f"attention_mask={getattr(attention_mask, 'dtype', None)} "
                f"position_ids={getattr(position_ids, 'dtype', None)} "
                f"occ_enabled={(bool(getattr(self.config, 'enable_occ_geom_loss', False)) or bool(getattr(self.config, 'enable_occ_temp_loss', False)))}"
            )
        inputs_embeds = inputs_embeds.to(self.model.embed_tokens.weight.dtype)
        if self._dtype_debug_should_log():
            rank_print(
                "[DTYPE_DEBUG][predict_box][aligned before self.model] "
                f"inputs_embeds={getattr(inputs_embeds, 'dtype', None)} "
                f"embed_tokens_weight={self.model.embed_tokens.weight.dtype}"
            )
        disable_lm_compile = os.getenv("ROSS3D_DISABLE_LM_COMPILE", "0") == "1"
        disable_flash_attn = os.getenv("ROSS3D_DISABLE_FLASH_ATTN", "0") == "1"
        prev_attn_impl = self._get_lm_attn_impl()
        if disable_flash_attn:
            self._set_lm_attn_impl("sdpa")
        if self._should_log_rank0():
            rank_print(
                "[NAN_DEBUG][predict_box][lm_call] "
                f"compile_mode={'eager' if disable_lm_compile else 'compiled'} "
                f"attn_impl={self._get_lm_attn_impl()} disable_flash_attn={disable_flash_attn}"
            )

        lm_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        self._retain_lm_boundary_grad("lm_inputs_embeds", inputs_embeds)
        try:
            if disable_lm_compile:
                outputs = self._run_lm_eager(**lm_kwargs)
            else:
                outputs = self.model(**lm_kwargs)
        finally:
            if disable_flash_attn and (prev_attn_impl is not None):
                self._set_lm_attn_impl(prev_attn_impl)

        hidden_states = outputs[0]
        self._retain_lm_boundary_grad("lm_hidden_states", hidden_states)
        if self._dtype_debug_should_log():
            rank_print(
                "[DTYPE_DEBUG][predict_box][after self.model] "
                f"hidden_states={hidden_states.dtype}"
            )

        ground_locations = (labels >= self.config.ground_token_ids[0]) & (labels <= self.config.ground_token_ids[-1])
        ground_hidden = hidden_states[ground_locations].squeeze(1)
        self._nan_debug_tensor_stats("ground_hidden", ground_hidden)
        self._retain_lm_boundary_grad("ground_hidden", ground_hidden)

        hook_state = {"fired": False}

        def _register_nonfinite_grad_hook(name: str, tensor: Optional[torch.Tensor]):
            if tensor is None or (not torch.is_tensor(tensor)) or (not tensor.requires_grad):
                return

            def _hook(grad):
                if hook_state["fired"]:
                    return grad
                if grad is None:
                    return grad
                if not torch.isfinite(grad).all():
                    hook_state["fired"] = True
                    if os.getenv("ROSS3D_NAN_DEBUG", "0") == "1":
                        if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                            rank_print(
                                "[NAN_DEBUG][predict_box][backward] "
                                f"first_nonfinite_grad_tensor={name} "
                                f"grad_shape={tuple(grad.shape)} grad_dtype={grad.dtype}"
                            )
                return grad

            tensor.register_hook(_hook)

        _register_nonfinite_grad_hook("ground_hidden", ground_hidden)
        
        if self.ground_head_type == 'mlp':
            ground_hidden = self.ground_head(ground_hidden).squeeze(0) 
            scores = (ground_hidden * object_features).sum(dim=-1)
            self._nan_debug_tensor_stats("scores_mlp", scores)
            self._retain_lm_boundary_grad("lm_scores", scores)
            _register_nonfinite_grad_hook("scores_mlp", scores)
        elif self.ground_head_type == 'score':
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype)) # B, C
            query_feat = self.ground_head_query(ground_hidden) # 1, C
            self._nan_debug_tensor_stats("obj_feat_score", obj_feat)
            self._nan_debug_tensor_stats("query_feat_score", query_feat)
            _register_nonfinite_grad_hook("obj_feat_score", obj_feat)
            _register_nonfinite_grad_hook("query_feat_score", query_feat)
            # sim = (F.normalize(obj_feat) * F.normalize(query_feat)).sum(dim=-1)
            mul_feat = obj_feat * query_feat
            scores = self.ground_head_score(mul_feat) # B, 1
            scores = scores.squeeze(1)
            self._nan_debug_tensor_stats("scores_score", scores)
            self._retain_lm_boundary_grad("lm_scores", scores)
            _register_nonfinite_grad_hook("scores_score", scores)

        elif self.ground_head_type == "infonce":
            object_features = torch.cat([object_features, self.ground_head_zero_target.unsqueeze(0)], dim=0)
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype))
            query_feat = self.ground_head_query(ground_hidden)
            obj_feat_norm = obj_feat.norm(dim=-1)
            query_feat_norm = query_feat.norm(dim=-1)
            self._nan_debug_tensor_stats("obj_feat_infonce_pre_norm", obj_feat)
            self._nan_debug_tensor_stats("query_feat_infonce_pre_norm", query_feat)
            self._nan_debug_tensor_stats("obj_feat_infonce_pre_norm_norm", obj_feat_norm)
            self._nan_debug_tensor_stats("query_feat_infonce_pre_norm_norm", query_feat_norm)
            _register_nonfinite_grad_hook("obj_feat_infonce_pre_norm", obj_feat)
            _register_nonfinite_grad_hook("query_feat_infonce_pre_norm", query_feat)
            obj_feat = F.normalize(obj_feat)
            query_feat = F.normalize(query_feat)
            scores = (obj_feat * query_feat).sum(dim=-1)
            self._nan_debug_tensor_stats("scores_infonce", scores)
            self._retain_lm_boundary_grad("lm_scores", scores)
            _register_nonfinite_grad_hook("scores_infonce", scores)

        loss = hidden_states.new_zeros(())
        lm_loss = hidden_states.new_zeros(())
        if box_labels is not None:
            if self.ground_head_type == "infonce":
                if len(box_labels[0]) == 0: # zero-target
                    box_labels[0].append(-1)
                scaled_scores = scores / self.ground_head_temperature
                self._nan_debug_tensor_stats("scaled_scores_infonce", scaled_scores)
                logits = torch.exp(scaled_scores)
                self._nan_debug_tensor_stats("logits_infonce", logits)
                pos_sum = logits[box_labels[0]].sum()
                all_sum = logits.sum()
                self._nan_debug_tensor_stats("logits_pos_sum_infonce", pos_sum)
                self._nan_debug_tensor_stats("logits_all_sum_infonce", all_sum)
                loss = - torch.log(pos_sum / all_sum)
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # for idx in box_labels[0]:
                #     loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                # loss /= len(box_labels[0])
            else:
                bce_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                target = torch.zeros_like(scores)
                target[box_labels[0]] = 1
                weight = torch.ones_like(scores)
                if len(box_labels[0]) != 0:
                    weight[box_labels[0]] *= (scores.shape[0] - len(box_labels[0])) / len(box_labels[0])

                bce_loss = (bce_loss_fct(scores, target.detach()) * weight).mean()
                loss = bce_loss
                # nce_loss = 0
                # logits = torch.exp(sim / self.ground_head_temperature)
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # if len(box_labels[0]) != 0:
                #     for idx in box_labels[0]:
                #         nce_loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                #     nce_loss /= len(box_labels[0])
                # loss = bce_loss + nce_loss

            lm_loss = loss.detach().clone()

        vm_loss, bev_loss = None, None
        # if self.training and getattr(self.config, 'ross_enable', False):
        #     # vm_loss = self.compute_vm_loss_v2(images, hidden_states, boi_ids, eoi_ids, newline_ids, mask)
        #     vm_loss = self.compute_vm_loss(images, hidden_states, boi_ids, eoi_ids, newline_ids, mask)
        #     loss += vm_loss
        #     if getattr(self.config, 'ross_multi_task', False):
        #         bev_loss = self.compute_vm_loss_bev(images, hidden_states, boi_ids, eoi_ids, newline_ids,
        #                                             video_dict["bev_image"], mask)
        #         loss += bev_loss

        return CausalLMOutputWithPastRoss(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            vm_loss=vm_loss,
            scores=scores,
            bev_loss=bev_loss,
            occupancy_aux_outputs=None,
            occ_geom_loss=None,
            occ_temp_loss=None,
        )

        # loss = None
        # if box_labels is not None:
        #     ## BCE
        #     loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        #     target = torch.zeros_like(scores)
        #     target[box_labels[0]] = 1
        #     weight = torch.ones_like(scores)
        #     weight[box_labels[0]] *= scores.shape[0] - 1
        #     loss = (loss_fct(scores, target.detach()) * weight).mean()
        #     ## CE 
        #     # loss_fct = nn.CrossEntropyLoss()
        #     # loss = loss_fct(scores, box_labels[0]) / self.config.ground_loss_scale


AutoConfig.register("ross3d_qwen", Ross3DQwenConfig)
AutoModelForCausalLM.register(Ross3DQwenConfig, Ross3DQwenForCausalLM)
