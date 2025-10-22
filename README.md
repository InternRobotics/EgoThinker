# 🦜 EgoThinker

This repo is the official implementation of EgoThinker at NIPS 2025 (coming soon)

> **["Unveiling Egocentric Reasoning with
Spatio-Temporal CoT"](https://openreview.net/forum?id=P6G1Z6jkf3)**<br>
> [Baoqi Pei](https://scholar.google.com/citations?user=sTCkd54AAAAJ), [Yifei Huang](https://scholar.google.com/citations?user=RU8gNcgAAAAJ), [Jilan Xu](https://scholar.google.com/citations?user=mf2U64IAAAAJ), Yuping He, [Guo Chen](https://scholar.google.com/citations?user=lRj3moAAAAAJ),<br> 
> [Fei Wu](https://scholar.google.com/citations?user=XJLn4MYAAAAJ),  [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ), [Jiangmiao Pang](https://scholar.google.com/citations?user=ssSfKpAAAAAJ&hl=zh-CN&oi=ao)<br>


<a href='https://github.com/InternRobotics/EgoThinker'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://github.com/InternRobotics/EgoThinker"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a> &nbsp;
<a href="https://github.com/InternRobotics/EgoThinker"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a> &nbsp;

⭐️: We are also working on a updated version for **spatial understanding** and **embodied QA**, stay tuned! 


---

### Introduction

Egocentric video reasoning centers on an unobservable agent behind the camera who dynamically shapes the environment, requiring inference of hidden intentions and recognition of fine-grained interactions.
This core challenge limits current multimodal large language models (MLLMs), which excel at visible event reasoning but lack embodied, first-person understanding.
To bridge this gap, we introduce \textbf{EgoThinker}, a novel framework that endows MLLMs with robust egocentric reasoning capabilities through spatio-temporal chain-of-thought supervision and a two-stage learning curriculum. 
First, we introduce \textbf{EgoRe-5M}, a large-scale egocentric QA dataset constructed from 13M diverse egocentric video clips. 
This dataset features multi-minute segments annotated with detailed CoT rationales and dense hand–object grounding. Second, we employ SFT on EgoRe-5M to instill reasoning skills, followed by reinforcement fine-tuning (RFT) to further enhance spatio-temporal localization. 
Experimental results show that EgoThinker outperforms existing methods across multiple egocentric benchmarks, while achieving substantial improvements in fine-grained spatio-temporal localization tasks. 

<div align="center">
<img src="assets/teaser.jpg">
</div> 


### 📰 News

- **Coming Soon:** Paper, Dataset, SFT/RFT training code and evaluation code release.


### 🚀 Getting Started

1. Clone the Repository

```bash
git clone https://github.com/InternRobotics/EgoThinker
cd EgoThinker
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


### 🛠️ Method

#### coming soon

### 🤗 Feedback & Support

We welcome feedback and issues. Thank you for trying our EgoThinker!

---

### 📄 Acknowledgments

Our code is built projects:

* **Qwen-VL** — [https://github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
* **VideoChat-R1** — [https://github.com/OpenGVLab/VideoChat-R1](https://github.com/OpenGVLab/VideoChat-R1)
* **lmms-eval** — [https://github.com/EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

---
