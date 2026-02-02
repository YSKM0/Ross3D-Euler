#!/bin/bash

MID_RUN_NAME="llava-video-qwen2-7b-ross3d-debug"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
# we have ./debug/video3dllm_debug.yaml
# --mm_pixel_decoder ./checkpoints/FLUX.1-dev/vae \
# --cycle_filter_positive_depth
#     --pretrain_mm_inv_adapter ./checkpoints/mm_inv_projector.bin \
# --mm_inv_projector_type denoiser_vit3x \

set -x

export ROSS3D_DEBUG_PARAM_READY=1
export ROSS3D_DEBUG_CHECKPOINT=1
export ROSS3D_DEBUG_MEMORY_SUMMARY=1

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=20409 \
    \
    ross3d/train/train_3d.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --view_mask_ratio 0.0 \
    --view_mask_prob 0.0 \
    \
    --version qwen_1_5 \
    --data_path ./scripts/3d/train/video3dllm_223k.yaml \
    --image_folder data \
    --video_folder data \
    --embodiedscan_folder ./data/embodiedscan/ \
    --mm_tunable_parts "mm_language_model" \
    --vision_tower ./models/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${MID_RUN_NAME} \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --save_only_model \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 660 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 14 \
    --world_position_embedding_type avg-discrete-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy uniform \
    --frames_upbound 3 \
    --cycle_consist True \
    --cycle_feature_source llm \
    --cycle_geo_mode clamped \
    --use_3d_coordinate True \
    --cycle_debug_memory False \
    --verbose_logging False \
    --cycle_detach_hidden_states True \
    --cycle_debug_grad False \
    --cycle_debug_optimizer False \
    --adamw_use_foreach True \
    --cycle_filter_positive_depth False \
    --model_name_or_path ./models/llava-video-qwen2-7b-ross3d
