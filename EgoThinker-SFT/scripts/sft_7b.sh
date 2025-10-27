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

# /mnt/hwfile/opencompass/checkpoints/llm/hf_hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f
# /mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2.5-VL-7B-Instruct/
# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
#/mnt/inspurfs/HOD_t/peibaoqi/mllm_hf/Qwen3-VL-8B-Instruct/
llm=/mnt/inspurfs/HOD_t/peibaoqi/mllm_hf/Qwen2.5-VL-7B-Instruct/

# Using HuggingFace model ID
PARTITION='HOD_t'
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
NNODE=4
NUM_GPU=4
# Training hyperparameters
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=clevrer%50,ego4dlong%20,ego4dshort%50,egotaskqa%50,egotimeqa%30,how2short%5,k400%40,nextqa%40,perception%40,qaego4d%40,sharegpt4o,sharegpt4v_coco,sharegpt4v_llava,sharegpt4v_sam,ssv2,youcook2%40,videochatgpt%50,holo_under,agibot_under,egoplan,coco_300k%30,robovqa_under%50,gqa%40,refcoco%50,ego4dcot,holo_reason,agibot_reason,robovqa_reason%50,visor,clevr_r1%30

#datasets=visor
# datasets=holo_reason,holo_under

# Output configuration
run_name="qwen2vl-baseline"
output_dir=/mnt/inspurfs/HOD_t/peibaoqi/ckpt/Egothinker/1027_qwen2.5vl/


## add data: robovqa;agibotqa;egoplan;scannet/scanrefer/scan3d;where2place;

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
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
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --mm_projector_lr 1e-4 \
    --vision_tower_lr 4e-5 \
    --optim adamw_torch \
    --base_interval 1 \
    --video_max_frames 32 \
    --video_min_frames 4 \
    --video_max_frame_pixels 451584 \
    --video_min_frame_pixels 50176 \
    --weight_decay 0.1 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to tensorboard" \

# Launch training
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n8 \
    --gres=gpu:8 \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --kill-on-bad-exit=1 \
    bash torchrun.sh \
    --nnodes=8 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    qwenvl/train/train_qwen.py \
    ${args}