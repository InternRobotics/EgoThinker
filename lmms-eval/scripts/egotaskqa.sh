# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/.cache/huggingface"

srun -p HOD_t --gres=gpu:4 accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/mnt/inspurfs/HOD_t/peibaoqi/ckpt/qwen2vl_egothinker/\
    --tasks egotaskqa \
    --log_samples \
    --output_path results/qwen2vl_1027