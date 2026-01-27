#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export WANDB_API_KEY=your_key_here



export CUDA_VISIBLE_DEVICES=4,5,6,7

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
