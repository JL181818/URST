# URST

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-URST__MODEL-yellow)](https://huggingface.co/JL18/URST_MODEL)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-URST__DATA-yellow)](https://huggingface.co/JL18/URST_DATA)

This is the official repository for our paper: **Enhancing GUI Agent with Uncertainty-Aware Self-Trained Evaluator**.

This project is based on [EasyR1](https://github.com/hiyouga/EasyR1) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Requirements

### Hardware Requirements (Estimated)

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2*24GB | 4*40GB | 8*40GB | 16*80GB | 32*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1*24GB | 1*40GB | 4*40GB |  8*80GB | 16*80GB |

> [!NOTE]
> Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` to enable bf16 training.

## Installation & Usage

This project currently requires two separate environments for SFT and RL training stages.

### Stage 1: SFT (Supervised Fine-Tuning)

#### 1. Environment Setup (SFT)

```bash
# Create environment for SFT
conda create --name urst_sft python=3.10
conda activate urst_sft

# Install LLaMA-Factory dependencies
cd LLaMA-Factory
pip install -e .
cd ..
```

#### 2. Configuration
Modify /EasyR1/LLaMA-Factory/examples/train_full/qwen2_5_vl_evaluator_full_sft.yaml

```
model_name_or_path: /path/to/your/base/model  # e.g., Qwen/Qwen2.5-VL-3B-Instruct
output_dir: /path/to/save/sft/model           # e.g., ./checkpoints/sft_round_init
train_img_dir: /path/to/your/train_images
```


####  3. Run SFT Training


```bash
# Ensure you are in the project root (URST/EasyR1)
python ./verl/trainer/sft.py \
    --train_files /path/to/your/train_data.json \
    --project_name easyr1 \
    --experiment_name my_experiment
``` 


#### 4. Evaluate SFT Model
```
python verl/trainer/eval_sft.py --model_path  /path/to/save/sft/model
```

### Stage 2: RL Training (SGPO - 3 Rounds)

#### 1. Environment Setup (RL)
Note: This stage requires a different environment based on EasyR1/veRL.

```bash
conda deactivate
conda create --name urst_rl python=3.9
conda activate urst_rl

# Install EasyR1 dependencies
pip install -e .
```

#### 2.Configuration
Modify /URST/EasyR1/examples/config.yaml
```
data:
  val_files: /path/to/your/test_data.json
  image_dir: /path/to/your/test_images
  image_train_dir: /path/to/your/train_images

trainer:
  save_checkpoint_path: /path/to/save/rl/checkpoints
  load_checkpoint_path: /path/to/save/sft/model  # Path to the SFT model from Stage 1
```

#### Run SGPO Training

##### Modify EasyR1/examples/qwen2_5_vl_3b_urst_sgpo.sh

```
export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH=/path/to/your/model #SFT保存的checkpoint
CHECKPOINT_PATH=/path/to/checkpoint/saves/easy_r1/experiment_name/SGPO_models_v1

    data.train_files=/path/to/data/train_data/processed_candidate_data_labelled_list.json \

trainer.n_gpus_per_node=4 \ #the number of CUDA_VISIBLE_DEVICES

```
##### Run the script

```bash

bash ./examples/qwen2_5_vl_3b_urst_sgpo.sh

```

#### Evaluation
##### Modify EasyR1/examples/test.sh

```bash
# Define paths (Please modify these paths to match your environment)

CONFIG_PATH="examples/config.yaml"

# Path to the tokenizer/processor (usually the SFT model path)

TOKENIZER_PATH="/share/urst_data/saves/easy_r1/qwen2_5_vl_3b_urst_sgpo_para_from_code/SFT_models/round_init"

# Base directory for test data and images

TEST_DATA_DIR="/share/urst_data/data/test_data"

# Path to the format prompt jinja file

FORMAT_PROMPT_PATH="examples/format_prompt/urst.jinja"

# Directory to save evaluation results

OUTPUT_DIR="output/eval_results"
```
##### Run the evaluation
```bash
bash ./examples/test.sh
```