#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# 非交互式登录所需环境变量（替换成你的真实值） 
export WANDB_API_KEY=your_key_here



# 选用空闲卡（根据你的 nvidia-smi，GPU1/2 空闲）
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Fix: 2025-12-03
# 现象: 训练初始化时 NCCL Timeout，显存没爆，内存爆了/死锁。
# 原因: L40S 没有 NVLink，默认 P2P 通信在 PCIe 上卡死。
# 作用: 强制 NCCL 仅在有 NVLink 时才用 P2P，否则降级走内存。
export NCCL_P2P_LEVEL=NVL

MODEL_PATH=/path/to/your/model
CHECKPOINT_PATH=/path/to/checkpoint/saves/easy_r1/experiment_name/SGPO_models_v7

# TODO: Update rollout_batch_size, n_gpus_per_node, global_batch_size, sample_sizes as needed


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/path/to/data/train_data/processed_candidate_data_labelled_list.json \
    data.format_prompt=/path/to/URST/EasyR1/examples/format_prompt/urst.jinja \
    data.prompt_key='prompt' \
    data.answer_key='gt' \
    data.image_key='image_paths' \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_urst_sgpo \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=${CHECKPOINT_PATH}
