# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration

from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video_QA as Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from src.open_r1.my_qwen_utils import process_vision_info
from tqdm import tqdm
import torch
import json
import random
import ast

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False


def calculate_t_iou(interval1, interval2):
    """
    计算两个时间区间的 IOU（交并比）
    
    参数:
        interval1: list [s1, e1]，第一个时间区间
        interval2: list [s2, e2]，第二个时间区间
    
    返回:
        float: IOU 值（范围 [0, 1]），无重叠时返回 0
    """
    s1, e1 = interval1
    s2, e2 = interval2
    
    # 计算交集
    intersection_start = max(s1, s2)
    intersection_end = min(e1, e2)
    intersection = max(0, intersection_end - intersection_start)  # 无重叠时为 0
    
    # 计算并集
    union = (e1 - s1) + (e2 - s2) - intersection
    
    # 避免除以 0（如果两个区间长度都为 0）
    if union == 0:
        return 0.0
    
    return intersection / union

import re

def validate_and_extract_tg(input_str):
    """
    验证字符串是否符合 (s,e) 格式，并提取其中的 2 个浮点数。
    
    参数:
        input_str (str): 输入的字符串
        
    返回:
        tuple: 如果验证成功，返回 (s, e) 两个浮点数
        None: 如果验证失败
    """
    # 定义正则表达式匹配模式
    pattern = r'^\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$'
    
    match = re.fullmatch(pattern, input_str.strip())
    if not match:
        return None
    
    try:
        # 提取并转换数字
        s = float(match.group(1))
        e = float(match.group(2))
        
        return (s, e)
    except ValueError:
        return None


def calculate_iou(box1, box2):
    """
    计算两个归一化边界框的交并比（IoU）。
    
    参数:
        box1 (list or tuple): 第一个边界框，[x_min, y_min, x_max, y_max]，值归一化到 [0, 1]。
        box2 (list or tuple): 第二个边界框，[x_min, y_min, x_max, y_max]，值归一化到 [0, 1]。
    
    返回:
        float: 两个边界框的IoU值，范围 [0, 1]。
    """
    # 获取交集矩形的边界
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    
    # 计算交集的宽度和高度
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    
    # 交集面积
    inter_area = inter_width * inter_height
    
    # 计算两个边界框的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = area_box1 + area_box2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

import re

def validate_and_extract(input_str):
    """
    验证字符串是否符合 [(x1,x2),(y1,y2)] 格式，并提取其中的 4 个数字。
    
    参数:
        input_str (str): 输入的字符串
        
    返回:
        tuple: 如果验证成功，返回 (x1, x2, y1, y2)，其中 x1, x2 是浮点数，y1, y2 可能是整数或浮点数
        None: 如果验证失败
    """
    # 定义正则表达式匹配模式
    pattern = r'^\[\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*,\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*\]$'
    
    match = re.fullmatch(pattern, input_str.strip())
    if not match:
        return None
    
    try:
        # 提取并转换数字
        x1 = float(match.group(1))
        x2 = float(match.group(2))
        y1 = float(match.group(3))  # 可以是整数或浮点数
        y2 = float(match.group(4))  # 可以是整数或浮点数
        
        return (x1, x2, y1, y2)
    except ValueError:
        return None


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou","format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="/share/wy/Video/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    # preprocessed_data_path: Optional[str] = field( # Add preprocessed_data_path argument
    #     default="",
    #     metadata={"help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."},
    # )

def validate_and_extract(s):
    # 使用正则表达式匹配格式
    pattern = r'^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$'
    match = re.fullmatch(pattern, s.strip())
    
    if match:
        # 提取匹配的四个整数
        a, b, c, d = map(int, match.groups())
        return (a, b, c, d)
    else:
        return None

def t_iou_glue_reward(completions,start,end,type, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    # print(completions, solution, durations)
    # contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, s, e, t in zip(completions, start, end,type):
        reward = 0.0
        if t == 'visor':
            rewards.append(reward)
            continue
        ground_gt = [s,e]
        pattern_glue = r'<answer>(.*?)</answer>'
        match_glue = re.search(pattern_glue, content, re.DOTALL)
        if match_glue:
            glue = match_glue.group(1)
            if validate_and_extract_tg(glue):
                s1,s2 = validate_and_extract_tg(glue)
                reward = calculate_t_iou([s1,s2], ground_gt)
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


def answer_reward(completions, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    def extract_characters_regex(s):
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is",
            "The correct option is",
            "Best answer:" "Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
            return ""

        matches = re.search(r"[ABCDEFG]", s)
        if matches is None:
            return ""
        return matches[0]
    
    rewards = []

    for content, sol in zip(completions, solution): 
        reward = 0.0
        
        pattern_answer = r'<answer>(.*?)</answer>'

        # 使用 search 方法查找首个匹配项
        match_answer = re.search(pattern_answer, content, re.DOTALL)

        if match_answer:
            # 获取捕获组中的内容
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(sol['answer']):
                reward = 1.0

        rewards.append(reward)

    return rewards

def iou_glue_reward(completions, gt,type, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []

    # print(completions, solution, durations, **kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, g, t in zip(completions, gt, type): # Added video_durations
        reward = 0.0

        if t == 'egoexo':
            rewards.append(reward)
            continue
        ground_gt = g
        pattern_glue = r'<answer>(.*?)</answer>'
        match_glue = re.search(pattern_glue, content, re.DOTALL)
        if match_glue:
            glue = match_glue.group(1)
            if validate_and_extract(glue):
                x1,y1,x2,y2 = validate_and_extract(glue)
                reward = calculate_iou([x1,y1,x2,y2], ground_gt)
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "iou": iou_glue_reward,
    "format": format_reward,
    "t_iou": t_iou_glue_reward
}



def load_json_dataset(train_data_path, eval_data_path, video_folder):#, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_paths, split_name):
        examples = []
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                data = data[::2]
            if 'egoexo' in file_path:
                for info in data:
                    video_path = info['vid']

                    example = {
                        "gt": [0,0,0,0],
                        "object": 'none',
                        "question":info['question'],
                        "video_path": video_path,
                        "start": info['start'],
                        "end": info['end'],
                        "type": "egoexo",
                    }
                    examples.append(example)
            elif 'visor' in file_path:
                for info in data:
                    
                    video_path = info['path']
                    example = {
                        "gt": info['gt'],
                        "object": info['object'],
                        "video_path": video_path,
                        "question":'none',
                        "start": -1,
                        "end": -1,
                        "type": "visor",
                    }
                    examples.append(example)
                    
        random.shuffle(examples)
        print(len(examples))
        dataset = Dataset.from_list(examples)

        #dataset.client = Client('~/petreloss.conf')
        dataset.client = None

        def __getitem__(self, idx): # Define getitem within the scope where dataset is available

            example = dataset[idx]
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset

            # try:
            if example['type'][0] == 'visor':

                messages = [{"role": "user", "content": [{"type": "image", "image": example["video_path"][0]}]}]
                image_inputs, video_inputs = process_vision_info([messages])
                # # data_to_return["image_inputs"] = [torch.load(os.path.join(example["video_path"][0], "image_inputs.pt"))]
                data_to_return["image_inputs"] = [image_inputs]
                data_to_return["type"] = ['visor']
            
            elif example['type'][0] == 'egoexo':

                messages = [{"role": "user", "content": [{"type": "video", "video": example["video_path"][0], "total_pixels": 1792 * 28 * 28, "min_pixels": 16 * 28 * 28, "fps":1.0}]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, client=self.client)
                fps_inputs = video_kwargs['fps']
                # # data_to_return["image_inputs"] = [torch.load(os.path.join(example["video_path"][0], "image_inputs.pt"))]
                data_to_return["video_inputs"] = [video_inputs]
                # with open(os.path.join(example["video_path"][0], "video_kwargs.json"), 'r') as f:
                data_to_return["video_kwargs"] = [video_kwargs]   
                data_to_return["type"] = ['egoexo']    

            # except Exception as e:
            #     print(f"Warning: Error loading preprocessed data from {example['video_path'][0]}, falling back to video_path. Error: {e}")

            #     print(idx)
            #     idx = idx + 1
            #     return self.__getitem__(idx)

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset

        return dataset

    

    train_dataset = create_dataset_from_json([train_data_path], "train")

    print(train_dataset[0])
    eval_dataset = create_dataset_from_json([train_data_path], "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset, now handles both raw and preprocessed data
    dataset = load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.video_folder,
        # script_args.preprocessed_data_path # Pass preprocessed_data_path
    )


    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError
    
    print("using: ", trainer_cls)

    # from peft import LoraConfig, get_peft_model

    # lora_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     inference_mode=False,
    #     r=64,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    # )

    # Initialize the GRPO trainer

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)