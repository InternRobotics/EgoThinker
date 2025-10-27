# Evaluation for egocentric and embodi tasks

We employ [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for model evaluation.

## Installation

Refer to the [original repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)  for installation instructions.

## Preparatory Steps
- Preparation of the test set JSON: Download the JSON file(s) to be evaluated and place them in the `anno` folder.

- Preparation of the test videos: Please check the `utils.py` file under each test task in `./lmms_eval/tasks`. Some tasks include a DATA_LIST entry, which means you’ll need to manually specify the directory where your local test videos are stored.

## Run

```python
bash scripts/egotaskqa.sh
```

## Comments

- The benchmark code for more Egocentric and Embodied QA benchmarks will be updated gradually. Please stay tuned for updates.

- Currently supported benchmarks: `EgotaskQA`, `VLN-QA`, `VISOR`