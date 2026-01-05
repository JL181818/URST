#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# 非交互式登录所需环境变量（替换成你的真实值） 
export WANDB_API_KEY=your_key_here



# 选用空闲卡（根据你的 nvidia-smi，GPU1/2 空闲）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_P2P_LEVEL=NVL

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

# Checkpoints to evaluate
# Example: "/path/to/ckpt/round_0/actor"
CHECKPOINTS=(
    "/share/urst_data/saves/easy_r1/qwen2_5_vl_3b_urst_sgpo_para_from_code/SGPO_models_v7/round_0/global_step_12/actor"
    "/share/urst_data/saves/easy_r1/qwen2_5_vl_3b_urst_sgpo_para_from_code/SGPO_models_v7/round_1/global_step_12/actor"
    "/share/urst_data/saves/easy_r1/qwen2_5_vl_3b_urst_sgpo_para_from_code/SGPO_models_v7/round_2/global_step_12/actor"
)
CHECKPOINT_NAMES=("round_0" "round_1" "round_2")

# Test files (JSON files located in TEST_DATA_DIR)
TEST_FILES=(
    "AITW_ID_traj_list.json"
    "AITW_OOD_traj_list.json"
    "AW_OOD_traj_list.json"
)

python3 -m verl.trainer.test \
    --config_path "${CONFIG_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --checkpoint_names "${CHECKPOINT_NAMES[@]}" \
    --test_data_dir "${TEST_DATA_DIR}" \
    --test_files "${TEST_FILES[@]}" \
    --format_prompt_path "${FORMAT_PROMPT_PATH}" \
    --output_dir "${OUTPUT_DIR}"