#!/bin/bash
set -euo pipefail

MID_RUN_NAME="ross3d-train-4"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Slurm context (these should exist inside the allocation step)
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NNODES=${SLURM_NNODES:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_PROCID=${SLURM_PROCID:-unset}  SLURM_LOCALID=${SLURM_LOCALID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set -x

# Rank-0 node hostname becomes the rendezvous master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Pick a port; keep your default
MASTER_PORT="${MASTER_PORT:-20409}"

# For a 2-node job with --ntasks-per-node=1, Slurm starts 2 tasks:
#   SLURM_PROCID = 0 on node 0, 1 on node 1
# We use that as the torchrun node_rank.
NODE_RANK="${SLURM_NODEID}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Required for multi-node: SLURM_NNODES must be set
: "${SLURM_NNODES:?SLURM_NNODES is not set. Are you running under srun/sbatch?}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is not set. Are you running under srun/sbatch?}"

echo "=== DISTRIBUTED LAUNCH CHECKS ==="
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Expect world size = $((SLURM_NNODES * NPROC_PER_NODE))"
echo "This node rank = $NODE_RANK"
echo "Visible GPUs (local): $CUDA_VISIBLE_DEVICES"
echo "==============================="

# export ROSS3D_DEBUG_CHECKPOINT_CALLS=1
# export ROSS3D_DEBUG_CHECKPOINT_CALLS_LIMIT=20

#  --torch_compile True \
#  --torch_compile_backend "inductor" \
# Launch distributed training: 4 ranks per node, across SLURM_NNODES nodes
python -m torch.distributed.run \
  --nproc_per_node=4 \
  --nnodes="${SLURM_NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  ross3d/train/train_3d.py \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --view_mask_ratio 0.25 \
  --view_mask_prob 0.25 \
  --deepspeed ./scripts/3d/zero3_original.json \
  --model_name_or_path ./checkpoints/LLaVA-Video-7B-Qwen2 \
  --pretrain_mm_inv_adapter ./checkpoints/mm_inv_projector.bin \
  --version qwen_1_5 \
  --data_path ./scripts/3d/train/video3dllm_223k.yaml \
  --image_folder data \
  --video_folder data \
  --embodiedscan_folder ./data/embodiedscan/ \
  --mm_tunable_parts "mm_mlp_adapter,mm_language_model,mm_inv_adapter" \
  --vision_tower ./models/siglip-so400m-patch14-384 \
  --ross_multi_task True \
  --mm_pixel_decoder ./checkpoints/FLUX.1-dev/vae \
  --mm_projector_type mlp2x_gelu \
  --mm_inv_projector_type denoiser_vit3x \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio anyres_max_9 \
  --image_grid_pinpoints "(1x1),...,(6x6)" \
  --mm_patch_merge_type spatial_unpad \
  --bf16 True \
  --run_name "${MID_RUN_NAME}" \
  --output_dir "./checkpoints/${MID_RUN_NAME}" \
  --num_train_epochs 1 \
  --per_device_eval_batch_size 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 1 \
  --weight_decay 0. \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 32768 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 \
  --lazy_preprocess True \
  --torch_compile True \
  --torch_compile_backend "inductor" \
  --dataloader_drop_last True \
  --mm_newline_position grid \
  --add_spatial_instruction True \
  --force_sample True \
  --mm_spatial_pool_stride 2 \
  --world_position_embedding_type avg-discrete-sin3d \
  --object_feature_type patch14-pe \
  --ground_head_type infonce \
  --group_by_task_length True \
  --frame_sampling_strategy uniform \
  --frames_upbound 32 \
  --verbose_logging False \
  --cycle_debug_grad False \
  --cycle_debug_optimizer False \
  --report_to wandb 