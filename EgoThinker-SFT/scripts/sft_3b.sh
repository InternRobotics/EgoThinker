export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_PORT=$((10086 + $RANDOM % 100))

export NCCL_ALGO=Tree

echo $MASTER_NODE
echo "All nodes used:"
echo ${ALL_NODES}
echo "Master node:"
echo ${MASTER_NODE}
echo "Args:"
echo $@

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=mllm_hf/Qwen2.5-VL-3B-Instruct/

# Using HuggingFace model ID
PARTITION='p'
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
NNODE=1
NUM_GPU=8
# Training hyperparameters
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=clevrer%20,ego4dcot,ego4dlong%15,ego4dshort,egotaskqa,egotimeqa%30,how2short%15,k400%30,nextqa,perception,qaego4d,sharegpt4o,sharegpt4v_coco,sharegpt4v_llava,ssv2,textcaps,youcook2,videochatgpt%30

# Output configuration
run_name="qwen2vl-baseline"
output_dir=qwen-vl-finetune/qwenvl_3b

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_pixels 451584 \
    --min_pixels 50176 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 12000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --base_interval 1 \
    --video_max_frames 32 \
    --video_min_frames 4 \
    --video_max_frame_pixels 200704 \
    --video_min_frame_pixels 37632 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to tensorboard" \

# Launch training
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n2 \
    --gres=gpu:8 \
    --async -o 'log/sftv2_3b' \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --kill-on-bad-exit=1 \
    bash torchrun.sh \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    qwenvl/train/train_qwen.py \
    ${args}