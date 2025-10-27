export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

DATA_VERSION="data/stage1_init_connector_iv1m.yaml"
DATA_VERSION_CLEAN=$(basename "$DATA_VERSION")
VISION_MODEL_VERSION="internvideo2"
VISION_MODEL_VERSION_CLEAN="internvideo2"

LLM_VERSION="Qwen/Qwen2_5-7B-Instruct"
LLM_VERSION_CLEAN="Qwen2_5_7B"

mm_projector_type=tome16_mlp_hd64
PROMPT_VERSION=plain

BASE_RUN_NAME=stage1-${VISION_MODEL_VERSION}-${mm_projector_type}-${LLM_VERSION_CLEAN}_${DATA_VERSION_CLEAN}_${PROMPT_VERSION}_$(date +"%Y%m%d_%H%M%S")
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"


PARTITION='video'
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
NUM_GPU=8
# NOTE: If you don't use slurm, please ref to https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/pretrain_clip.sh for training command.
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --ntasks=${NUM_GPU} \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python -u llava/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/stage1-init_connector/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    --frames_upbound 4 \
    --time_msg short \
    --local_num_frames 4 \
    --sample_type middle \
    --vision_encode_type video_image \
    --mm_pos_num_frames 4 \
    --mm_local_num_frames 4 \
    --verbose_logging True >> ./output_logs/stage1-init_connector/${BASE_RUN_NAME}.log
# You can delete the sdpa attn_implementation if you want to use flash attn

