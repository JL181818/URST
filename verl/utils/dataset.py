# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import Dataset as HFDataset, load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
from termcolor import cprint


def _detect_dataset_files(directory: str) -> tuple[str, list[str]]:
    """Return dataset file type and sorted list of JSON/JSONL files in directory."""
    json_files = []
    for entry in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(entry)[-1].lower()
        if ext in (".json", ".jsonl"):
            json_files.append(full_path)

    if not json_files:
        raise FileNotFoundError(f"No JSON/JSONL files found under {directory}")

    file_type = os.path.splitext(json_files[0])[-1][1:].replace("jsonl", "json")
    return file_type, json_files


def _load_local_json_dataset(json_files: list[str]) -> HFDataset:
    """Fallback loader for plain JSON dict/list files that HF builder can't parse."""
    records: list[dict[str, Any]] = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        if isinstance(content, dict):
            records.extend(content.values())
        elif isinstance(content, list):
            records.extend(content)
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}: {type(content)}")
    return HFDataset.from_list(records)


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "gt",
        image_key: str = "image_paths",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = False,
        filter_overlong_prompts_workers: int = 16,
        episode_id_key: str = "episode_id",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.episode_id_key = episode_id_key

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            file_type, json_files = _detect_dataset_files(data_path)
            try:
                self.dataset = load_dataset(file_type, data_files=json_files, split=data_split)
            except Exception as exc:
                print(f"[RLHFDataset] Falling back to manual JSON loading due to: {exc}")
                self.dataset = _load_local_json_dataset(json_files)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _make_prompt_text(self, example: dict[str, Any]) -> str:
        """Construct prompt text from new schema, falling back to legacy prompt if provided."""
        potential_prompt_keys = {
            self.prompt_key,
            str(self.prompt_key or "").lower(),
            str(self.prompt_key or "").upper(),
            "prompt",
            "Prompt",
        }
        sys_mesg1="You are an expert in evaluating the performance of an Android navigation agent. The agent is designed to help a human user navigate the device to complete a task. Given the user's intent, the agent's action history, and the last two states of the screen, your goal is to decide whether the agent has successfully completed the task.*Output Format* The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>reasoning process here</think><answer>answer here YES or NO</answer>."
        for key in potential_prompt_keys:
            value = example.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return sys_mesg1 + stripped

        user_intent = (example.get("User Intent") or "").strip()
        action_hist = (example.get("Action History") or "").strip()

        has_structured_fields = bool(user_intent or action_hist)
        if has_structured_fields:
            prompt_text = (
                f"User Intent: {user_intent}\n\n"
                f"Action History: {action_hist}\n\n"
                "the second last state and the last state of the screen are shown in the images."
            )
        else:
            prompt_text = (example.get(self.prompt_key) or "").strip()

        if has_structured_fields and self.format_prompt:
            # 可选：用外部模板覆盖（模板内请使用 {{ user_intent }} 与 {{ action_history }}）
            try:
                tmpl = Template(self.format_prompt.strip())
                prompt_text = tmpl.render(user_intent=user_intent, action_history=action_hist)
            except Exception:
                # 模板异常则回退到默认 prompt_text
                pass
        return prompt_text

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = self._make_prompt_text(example)
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            #    content_list = []
            # for i, content in enumerate(prompt_str.split("<image>")):
            #     if i != 0:
            #         content_list.append({"type": "image"})
            #     if content:
            #         content_list.append({"type": "text", "text": content}) """
            content = [{"type": "text", "text": prompt_str}]
            if len(example[self.image_key]) >= 2:
                content.append({"type": "image"})
                content.append({"type": "image"})
            elif len(example[self.image_key]) == 1:
                content.append({"type": "image"})
                # cprint(("data only have one image:", episode_id), "red")
            else:
                cprint(("data have no image:", example[self.episode_id_key]), "red")

            return [{"role": "user", "content": content}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]

            # 只取最后两张或最后一张图片
            selected_images = []
            if len(images) >= 2:
                selected_images = [images[-2], images[-1]]
            elif len(images) == 1:
                selected_images = [images[-1]]
            else:
                cprint(("data have no image:", example[self.episode_id_key]), "red")
                selected_images = []

            # 路径补全
            if self.image_dir is not None and len(selected_images) != 0 and isinstance(selected_images[0], str):
                selected_images = [os.path.join(self.image_dir, image) for image in selected_images]

            processed_images = [] if len(selected_images) != 0 else None  # text-only data
            for image in selected_images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        # 打印查看example的keys以及对应值
        # for key, value in example.items():
        #     print(f"{key}: {value}")

        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            # 只取最后两张或最后一张图片
            selected_images = []
            if len(images) >= 2:
                selected_images = [images[-2], images[-1]]
            elif len(images) == 1:
                selected_images = [images[-1]]
            else:
                cprint(("data have no image:", example.get(self.episode_id_key, "")), "red")
                selected_images = []

            # 路径补全
            if self.image_dir is not None and len(selected_images) != 0 and isinstance(selected_images[0], str):
                selected_images = [os.path.join(self.image_dir, image) for image in selected_images]
                print(f"[Dataset] Resolved image paths: {selected_images}")

            processed_images = [] if len(selected_images) != 0 else None  # text-only data
            for image in selected_images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": selected_images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example

class RLHF_train_Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor: Optional["ProcessorMixin"],
        prompt_key: str = "prompt",                  # kept for interface compatibility (ignored in new schema)
        answer_key: str = "gt",                      # kept for interface compatibility; we now prefer 'status'
        image_key: str = "image_paths",
        video_key: str = "videos",
        image_train_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,         # optional external template
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = False,
        filter_overlong_prompts_workers: int = 16,
        episode_id_key: str = "episode_id",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_train_dir = image_train_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.episode_id_key = episode_id_key

        # 支持三种 data_path 形式：本地目录、本地文件、HF Hub
        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            file_type, json_files = _detect_dataset_files(data_path)
            try:
                self.dataset = load_dataset(file_type, data_files=json_files, split=data_split)
            except Exception as exc:
                print(f"[RLHF_train_Dataset] Falling back to manual JSON loading due to: {exc}")
                self.dataset = _load_local_json_dataset(json_files)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        # 可选的外部模板（如果提供），你可以在模板里用 {{ user_intent }} 与 {{ action_history }} 两个变量
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    # ---------- helpers ----------

    def _make_prompt_text(self, example: dict[str, Any]) -> str:
        """Construct prompt text from new schema, falling back to legacy prompt if provided."""
        potential_prompt_keys = {
            self.prompt_key,
            str(self.prompt_key or "").lower(),
            str(self.prompt_key or "").upper(),
            "prompt",
            "Prompt",
        }
        for key in potential_prompt_keys:
            value = example.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped

        user_intent = (example.get("User Intent") or "").strip()
        action_hist = (example.get("Action History") or "").strip()

        has_structured_fields = bool(user_intent or action_hist)
        if has_structured_fields:
            prompt_text = (
                f"User Intent: {user_intent}\n\n"
                f"Action History: {action_hist}\n\n"
                "the second last state and the last state of the screen are shown in the images."
            )
        else:
            prompt_text = (example.get(self.prompt_key) or "").strip()

        if has_structured_fields and self.format_prompt:
            # 可选：用外部模板覆盖（模板内请使用 {{ user_intent }} 与 {{ action_history }}）
            try:
                tmpl = Template(self.format_prompt.strip())
                prompt_text = tmpl.render(user_intent=user_intent, action_history=action_hist)
            except Exception:
                # 模板异常则回退到默认 prompt_text
                pass
        return prompt_text

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """统一输出聊天消息结构；多模态时用 content = [{type:..., ...}, ...]。"""
        prompt_str: str = self._make_prompt_text(example)

        if self.image_key in example:
            # 走图像-文本路径
            content = [{"type": "text", "text": prompt_str}]
            if len(example[self.image_key]) >= 2:
                content.append({"type": "image"})
                content.append({"type": "image"})
            elif len(example[self.image_key]) == 1:
                content.append({"type": "image"})
            return [{"role": "user", "content": content}]
        elif self.video_key in example:
            # 走视频-文本路径（若存在）
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})
                if content:
                    content_list.append({"type": "text", "text": content})
            return [{"role": "user", "content": content_list}]
        else:
            # 文本-only
            return [{"role": "user", "content": prompt_str}]

    def _extract_ground_truth(self, example: dict[str, Any]) -> Any:
        """
        优先使用新字段：
            1) status
            2) 从 answer 文本中解析 "Status: XXX"
            3) 退回到 self.answer_key / 'gt'
        """
        # 3) 回退到旧字段（保持接口不变）
        for k in [self.answer_key, "gt", "GT", "label"]:
            if k in example and example[k] not in (None, ""):
                return example[k]

        # 1) 直取 status
        if "status" in example and example["status"] not in (None, ""):
            return example["status"]

        # 2) 从 answer 中抽取
        ans = example.get("answer")
        if isinstance(ans, str) and "Status:" in ans:
            import re
            m = re.search(r"Status:\s*([A-Za-z]+)", ans)
            if m:
                return m.group(1)

        

        # 实在没有，返回原 answer（可能是整段说明）
        return ans

    # ---------- length filter ----------

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]

            # 只取最后两张/最后一张
            if len(images) >= 2:
                selected_images = [images[-2], images[-1]]
            elif len(images) == 1:
                selected_images = [images[-1]]
            else:
                # 允许 text-only（但这里提示一下）
                cprint(("data have no image:", example.get(self.episode_id_key, example.get("id", ""))), "red")
                selected_images = []

            # 路径补全
            if self.image_train_dir is not None and selected_images and isinstance(selected_images[0], str):
                selected_images = [os.path.join(self.image_train_dir, image) for image in selected_images]

            processed_images = [] if selected_images else None  # text-only data
            if processed_images is not None:
                for image in selected_images:
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_train_dir is not None and videos and isinstance(videos[0], str):
                videos = [os.path.join(self.image_train_dir, v) for v in videos]

            processed_videos = [] if videos else None
            if processed_videos is not None:
                for v in videos:
                    processed_videos.append(process_video(v, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    # ---------- Dataset API ----------

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]

        # 构建 messages & prompt（新 schema 不再使用 self.prompt_key）
        messages = self._build_messages(example)
        # 在调用 processor 之前，打印 messages 的完整结构
        # cprint(("messages (raw):", messages), "red")

        # —— 多模态分支：图片 ——
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # 红色打印prompt
            # cprint(("prompt:", prompt), "red")
            images = example.pop(self.image_key)

            # 只取最后两张/最后一张
            if len(images) >= 2:
                selected_images = [images[-2], images[-1]]
            elif len(images) == 1:
                selected_images = [images[-1]]
            else:
                cprint(("data have no image:", example.get(self.episode_id_key, example.get("id", ""))), "red")
                selected_images = []

            # 路径补全
            if self.image_train_dir is not None and selected_images and isinstance(selected_images[0], str):
                selected_images = [os.path.join(self.image_train_dir, image) for image in selected_images]

            processed_images = [] if selected_images else None  # text-only data
            if processed_images is not None:
                for image in selected_images:
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": selected_images}

        # —— 多模态分支：视频 ——
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_train_dir is not None and videos and isinstance(videos[0], str):
                videos = [os.path.join(self.image_train_dir, v) for v in videos]

            processed_videos = [] if videos else None
            video_fps_list = []
            if processed_videos is not None:
                for v in videos:
                    processed_video, video_fps = process_video(
                        v, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                    )
                    processed_videos.append(processed_video)
                    video_fps_list.append(video_fps)

            model_inputs = self.processor(videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
            if "second_per_grid_ts" in getattr(self.processor, "model_input_names", []):
                model_inputs["second_per_grid_ts"] = [2.0 / fps for fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}

        # —— 纯文本分支 ——
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # 位置编码（保持与原版一致）
        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        # 截断/左填充等后处理
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # 计算 raw_prompt_ids（仅文本 token；用于训练时的参考/对齐）
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        # ground truth 归一化赋值
        gt_value = self._extract_ground_truth(example)

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = gt_value  # 训练阶段统一读取此字段

        return example
