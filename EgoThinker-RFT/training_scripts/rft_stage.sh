
# export WANDB_PROJECT=Video-GRPO
export NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${NAME}.log"


srun -p HOD_t \
    --job-name=${NAME} \
    -n1 \
    --gres=gpu:8 \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --kill-on-bad-exit=1 torchrun --nnodes=1 --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="10673" \
    src/open_r1/grpo_qa.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir /mnt/petrelfs/peibaoqi/robot/Egothinker/rft/egothinker/ \
    --model_name_or_path /mnt/inspurfs/HOD_t/peibaoqi/ckpt/Egothinker/qwen2.5_vl_rft/ \
    --train_data_path anno.visor.json \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --save_steps 1500 \
    --save_total_limit 1 \
    --save_only_model true

