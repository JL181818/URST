# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from typing import Optional

# import torch
# from torch.utils.data import RandomSampler, SequentialSampler
# from torchdata.stateful_dataloader import StatefulDataLoader
# from transformers import PreTrainedTokenizer, ProcessorMixin

# from ..utils.dataset import RLHFDataset, collate_fn, RLHF_train_Dataset
# from .config import DataConfig


# def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
#     train_dataset = RLHF_train_Dataset(
#         data_path=config.train_files,
#         tokenizer=tokenizer,
#         processor=processor,
#         prompt_key=config.prompt_key,
#         answer_key=config.answer_key,
#         image_key=config.image_key,
#         video_key=config.video_key,
#         image_train_dir=config.image_train_dir,
#         video_fps=config.video_fps,
#         max_prompt_length=config.max_prompt_length,
#         truncation="right",
#         format_prompt=config.format_prompt,
#         min_pixels=config.min_pixels,
#         max_pixels=config.max_pixels,
#         filter_overlong_prompts=config.filter_overlong_prompts,
#         filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
#     )
#     # use sampler for better ckpt resume
#     if config.shuffle:
#         train_dataloader_generator = torch.Generator()
#         train_dataloader_generator.manual_seed(config.seed)
#         sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
#     else:
#         sampler = SequentialSampler(data_source=train_dataset)

#     if config.mini_rollout_batch_size is not None:
#         train_batch_size = config.mini_rollout_batch_size
#     else:
#         train_batch_size = config.rollout_batch_size

#     train_dataloader = StatefulDataLoader(
#         dataset=train_dataset,
#         batch_size=train_batch_size,
#         sampler=sampler,
#         num_workers=8,
#         collate_fn=collate_fn,
#         pin_memory=False,
#         drop_last=True,
#     )

#     val_dataset = RLHFDataset(
#         data_path=config.val_files,
#         tokenizer=tokenizer,
#         processor=processor,
#         prompt_key=config.prompt_key,
#         answer_key=config.answer_key,
#         image_key=config.image_key,
#         video_key=config.video_key,
#         image_dir=config.image_dir,
#         video_fps=config.video_fps,
#         max_prompt_length=config.max_prompt_length,
#         truncation="right",
        
#         format_prompt=config.format_prompt,
#         min_pixels=config.min_pixels,
#         max_pixels=config.max_pixels,
#         filter_overlong_prompts=config.filter_overlong_prompts,
#     )

#     if config.val_batch_size == -1:
#         val_batch_size = len(val_dataset)
#     else:
#         val_batch_size = config.val_batch_size

#     val_dataloader = StatefulDataLoader(
#         dataset=val_dataset,
#         batch_size=val_batch_size,
#         shuffle=False,
#         num_workers=8,
#         collate_fn=collate_fn,
#         pin_memory=False,
#         drop_last=False,
#     )

    
#     # print(f"train_dataloader type: {type(train_dataloader)}")
#     # print(f"train_dataloader len: {len(train_dataloader)}")
#     # print(f"train_dataloader batch_size: {train_dataloader.batch_size}")
#     # print(f"train_dataloader sampler: {type(train_dataloader.sampler)}")
#     # print(f"train_dataloader dataset type: {type(train_dataloader.dataset)}")
#     # print(f"train_dataloader dataset len: {len(train_dataloader.dataset)}")
#     # print(f"val_dataloader len: {len(val_dataloader)}")
#     assert len(train_dataloader) >= 1
#     assert len(val_dataloader) >= 1
#     return train_dataloader, val_dataloader


# Copyright 2024 Bytedance Ltd. and/or its affiliates
# gemini
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

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
# Removed Sampler and DataLoader imports as they are no longer created here
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn, RLHF_train_Dataset
from .config import DataConfig


def create_datasets(
    config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]
) -> Tuple[Dataset, Dataset]:
    """
    Creates the training and validation Dataset objects from the config.
    
    This function is modified to return Dataset objects instead of DataLoaders,
    allowing the trainer to manage data loading internally. This is crucial
    for the URST active learning loop, where the training dataset is
    dynamically generated before each training (SGPO) phase.
    """
    
    train_dataset = RLHF_train_Dataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_train_dir=config.image_train_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    # Removed DataLoader creation logic
    
    # print(f"train_dataset type: {type(train_dataset)}")
    # print(f"train_dataset len: {len(train_dataset)}")
    # print(f"val_dataset type: {type(val_dataset)}")
    # print(f"val_dataset len: {len(val_dataset)}")
    
    assert len(train_dataset) >= 1, "Training dataset must not be empty"
    assert len(val_dataset) >= 1, "Validation dataset must not be empty"
    
    return train_dataset, val_dataset