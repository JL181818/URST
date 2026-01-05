"""Reward function for binary YES/NO tasks.

It parses the model output to extract an answer between `<answer>...</answer>`
tags (fallback to the raw string), normalizes to YES/NO, and assigns an
accuracy-style reward. UNKNOWN predictions get a configurable penalty.
"""

import re
from typing import Any


def _parse_answer(text: str) -> str:
    """Return normalized YES/NO/UNKNOWN from a model output string."""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = match.group(1) if match else text
    candidate = str(candidate).strip().upper()
    if "YES" in candidate:
        return "YES"
    if "NO" in candidate:
        return "NO"
    return "UNKNOWN"


FORMAT_PATTERN = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])", flags=re.DOTALL | re.MULTILINE)


def _format_reward(text: str) -> float:
    """Return 1.0 if the response follows the <think>/<answer> format exactly."""
    return 1.0 if FORMAT_PATTERN.match(text.strip()) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], unknown_penalty: float = 0.0) -> list[dict[str, float]]:
    """
    Batch reward function compatible with `reward_type=batch`.

    Each item in `reward_inputs` should contain:
      - response: model output string
      - ground_truth: expected label (YES/NO)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for yes_no reward function.")

    scores: list[dict[str, float]] = []
    # # 打印reward_inputs的形状和内容以进行调试
    # print(f"Debug: reward_inputs length = {len(reward_inputs)}")
    # print(f"Debug: reward_inputs content = {reward_inputs}")
    for item in reward_inputs:
        pred = _parse_answer(item["response"])
        gt = str(item["ground_truth"]).strip().upper()
        fmt_reward = _format_reward(item["response"])
        if pred == "UNKNOWN":
            acc = 0.0
            overall = unknown_penalty+fmt_reward
        else:
            acc = 1.0 if pred == gt else 0.0
            overall = acc + fmt_reward

        scores.append(
            {
                "overall": overall,
                "accuracy": acc,
                "format": fmt_reward,
            }
        )

    return scores
