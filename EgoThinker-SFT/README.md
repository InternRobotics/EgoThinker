# SFT Code for EgoThinker 

## Getting Started

1. Clone the Repository

```bash
git clone https://github.com/InternRobotics/EgoThinker
cd EgoThinker/EgoThinker-SFT
````

2. Set Up the Python Environment

```bash
conda create -n egothinker python=3.10.18
conda activate egothinker
```

3. Install project dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation
To prepare the training data, you need to download the annotations from Hugging Face and place them in the `qwenvl/anno` directory, and update the `root_path` for each dataset in `__init__.py`.

To add or modify datasets for training, follow these steps:

1. **Create a dataset dictionary** in the format in the file `data/__init__.py`:
```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/image/data",  # Can be empty if paths are in annotations
}
```

2. **Register your dataset** by adding it to the `data_dict`:
```python
data_dict = {
    "your_dataset_name": DATASET_NAME,
    # ... other datasets
}
```

### Sampling Rate Control

You can optionally specify sampling rates by appending `%X` to the dataset name:
- `"dataset_name%50"` will sample 50% of the data
- `"dataset_name%20"` will sample 20% of the data

### Usage Example

1. Define your dataset:
```python
MY_DATASET = {
    "annotation_path": "/data/my_dataset/annotations.json",
    "data_path": "/data/my_dataset/images/",
}

data_dict = {
    "my_dataset": MY_DATASET,
    "cambrian_737k": CAMBRIAN_737K,  # existing dataset
}
```

2. Use it in training:
```python
dataset_names = ["my_dataset%50"]  # Will use 50% of your dataset
configs = data_list(dataset_names)
```

## Training

```python
bash scripts/sft_7b.sh
```

We support Qwen2-VL and Qwen2.5-VL.
## Repository Structure

### `train/`
- `trainer.py`: Main trainer updated from Huggingface Trainer
- `train_qwen.py`: Main file for training
- `argument.py`: Dataclasses for model, data and training arguments

### `data/`
- `__init__.py`: Contains datasets configs
- `data_qwen.py`: Data processing module for QwenVL models
- `rope2d.py`: Provide RoPE implementation

### `tools`
- `process_bbox.ipynb`: Convert bbox into QwenVL format. If you have grounding data, please refer this file to tranform your data.
- `pack_data.py`: Pack data into even length buckets.

## Comments

Our SFT code was built from [Qwen-VL](https://github.com/QwenLM/Qwen3-VL), and if you have questions, you might find solutions in the original repository.