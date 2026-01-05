import argparse
import json
import os
import re
from collections import Counter

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm

# --- [ 1. START: LLaMA-Factory Registration Fix ] ---
import sys
try:
    from paths import LLAMAF_DIR
except ImportError:
    print(f"错误: 无法从 'paths.py' 导入 `LLAMAF_DIR`。")
    print("请确保 'paths.py' 文件与 `eval_sft.py` 位于同一目录（或在 Python 路径上）。")
    print(f"`sft.py` 也依赖这个文件。")
    sys.exit(1)

# 将 LLaMA-Factory 的 src 目录添加到 sys.path 中（真实的 Python 包位于此处）
llamaf_src = os.path.join(str(LLAMAF_DIR), "src")
if llamaf_src not in sys.path:
    sys.path.insert(0, llamaf_src)
    print(f"[INFO] 已将 LLaMA-Factory src 路径添加到 sys.path: {llamaf_src}")

try:
    import llamafactory
    print("[INFO] 成功导入 LLaMA-Factory (用于模型注册)。")
except ImportError:
    print(f"[ERROR] 无法导入 'llamafactory'。")
    print(f"请确保 LLaMA-Factory 已在 {LLAMAF_DIR} 正确安装 (例如: pip install -e .[vl])")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] 导入 llamafactory 时出错: {e}")
    sys.exit(1)

from transformers import AutoProcessor
try:
    # 尝试直接导入 Qwen2_5_VL 的专用类
    from transformers import Qwen2_5_VLForConditionalGeneration
    print("[INFO] 成功导入 Qwen2_5_VLForConditionalGeneration")
except ImportError:
    print("[WARNING] 无法导入 Qwen2_5_VLForConditionalGeneration，尝试使用 AutoModel...")
    from transformers import AutoModel as Qwen2_5_VLForConditionalGeneration

# 导入 Qwen-VL 的工具函数

from qwen_vl_utils import process_vision_info
print("[INFO] 成功导入 qwen_vl_utils")

#
# --- [ END: LLaMA-Factory Registration Fix ] ---


# --- [ 2. Helper Functions ] ---

SYSTEM_PROMPT = """You are an expert in evaluating the performance of an Android navigation agent. The agent is designed to help a human user navigate the device to complete a task. Given the user's intent, the agent's action history, and the last two states of the screen, your goal is to decide whether the agent has successfully completed the task or not.

*Output Format* The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>reasoning process here</think><answer>answer here YES or NO</answer>."""

def load_json(data_path):
    """加载一个 JSON 文件。"""
    with open(data_path, "r") as f:
        return json.load(f)

def parse_answer_from_output(model_output: str) -> str:
    """
    从模型的完整输出字符串中提取 "YES" 或 "NO" 答案。
    """
    # 优先搜索 <answer> 标签
    match = re.search(r"<answer>(.*?)</answer>", model_output, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip().upper()
        if "YES" in answer:
            return "YES"
        if "NO" in answer:
            return "NO"
    
    # 回退机制
    last_part = model_output.split("assistant")[-1].strip().upper()
    if last_part.endswith("YES"):
        return "YES"
    if last_part.endswith("NO"):
        return "NO"
    if last_part == "YES":
        return "YES"
    if last_part == "NO":
        return "NO"

    return "UNKNOWN"

# --- [ 3. 评估主函数 ] ---

def evaluate_model(args):
    """
    加载 SFT 模型并评估指定的测试集。
    """
    print(f"--- 1. 加载模型和 Processor ---")
    print(f"模型路径: {args.model_path}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES 已设置为: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print(f"[INFO] 可用 GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[INFO] 当前 GPU 设备: {torch.cuda.get_device_name(0)}")
    
    # 使用 AutoProcessor 而不是 AutoTokenizer
    processor = AutoProcessor.from_pretrained(args.sft_model_path, trust_remote_code=True)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # 官方API支持 torch_dtype，而不是 dtype，原本代码是dtype=torch.bfloat16
        device_map={"": 0},  # 强制使用第一个可见的 GPU（即 GPU 2）
        trust_remote_code=True
    )
    model.eval()
    print("模型和 Processor 加载完成。")

    # 定义要测试的文件列表
    test_files = [
        "AITW_ID_traj_list.json",
        "AITW_OOD_traj_list.json",
        "AW_OOD_traj_list.json",
    ]
    
    results = {} # 存储所有结果

    for test_file in test_files:
        test_file_path = os.path.join(args.test_data_dir, test_file)
        if not os.path.exists(test_file_path):
            print(f"警告：未找到测试文件 {test_file_path}，跳过。")
            continue
            
        print(f"\n--- 2. 开始评估: {test_file} ---")
        
        # 加载数据
        try:
            test_data = load_json(test_file_path)
            print(f"成功加载 {len(test_data)} 个样本。")
        except Exception as e:
            print(f"错误：加载 {test_file_path} 失败: {e}")
            continue

        predictions = []
        ground_truths = []

        # 逐个样本进行推理
        for item in tqdm(test_data, desc=f"评估 {test_file}"):
            try:
                # 1. 准备输入数据
                user_intent = item.get("prompt")
                action_history = item.get("Action History", "") 
                
                gt_label = item.get(args.answer_key, "UNKNOWN").upper()
                if gt_label not in ["YES", "NO"]:
                    continue

                # 2. 准备图像路径（取最后两张图片）
                image_paths_relative = item.get("image_paths", [])
                if len(image_paths_relative) < 2:
                    print(f"警告：样本 {item.get('id', 'N/A')} 图片数量不足 2 张，跳过。")
                    continue
                
                # 获取最后两张图片的完整路径
                img1_path = os.path.join(args.image_dir, image_paths_relative[-2])
                img2_path = os.path.join(args.image_dir, image_paths_relative[-1])
                
                # 3. 构建符合 Qwen2-VL 格式的消息（参考 utils.py 的逻辑）
                # 文本描述
                text_content = f"User Intent: {user_intent}\n\nAction History: {action_history}\n\nthe second last state and the last state of the screen are shown in the images."
                
                # 构建消息格式：先放图片，后放文本
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img1_path},
                            {"type": "image", "image": img2_path},
                            {"type": "text", "text": text_content}
                        ]
                    }
                ]
                
                # 4. 使用 processor 处理输入
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # 处理视觉信息
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Tokenize
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                # 5. 模型推理
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=150, 
                    do_sample=False 
                )
                
                # 6. 解码输出
                output_ids = gen_out[0][len(inputs["input_ids"][0]):]
                response = processor.decode(output_ids, skip_special_tokens=True)

                # 7. 解析输出
                pred_label = parse_answer_from_output(response)
                
                predictions.append(pred_label)
                ground_truths.append(gt_label)

            except Exception as e:
                print(f"错误：处理样本 {item.get('id', 'N/A')} 失败: {e}")
                import traceback
                traceback.print_exc()
                predictions.append("UNKNOWN")
                if 'gt_label' in locals():
                    ground_truths.append(gt_label)
                else:
                    continue 

        # --- 3. 计算指标 ---
        print(f"--- 评估完成: {test_file} ---")
        
        labels_order = ["YES", "NO", "UNKNOWN"]
        
        acc = accuracy_score(ground_truths, predictions)
        f1 = f1_score(ground_truths, predictions, labels=labels_order, average="weighted")
        pre = precision_score(ground_truths, predictions, labels=labels_order, average="weighted", zero_division=0)
        rec = recall_score(ground_truths, predictions, labels=labels_order, average="weighted", zero_division=0)
        
        binary_gt = [1 if g == "YES" else 0 for g in ground_truths if g in ["YES", "NO"]]
        binary_pred = [1 if p == "YES" else 0 for p, g in zip(predictions, ground_truths) if g in ["YES", "NO"]]

        if binary_gt:
            tn, fp, fn, tp = confusion_matrix(binary_gt, binary_pred, labels=[0, 1]).ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        num_unknown = predictions.count("UNKNOWN")
        
        results[test_file] = {
            "Accuracy": acc,
            "F1-Score (Weighted)": f1,
            "Precision (Weighted)": pre,
            "Recall (Weighted)": rec,
            "TP (YES)": tp,
            "FN (YES)": fn,
            "TN (NO)": tn,
            "FP (NO)": fp,
            "Unknowns": num_unknown,
            "Total": len(ground_truths)
        }

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (Weighted): {f1:.4f}")
        print(f"Precision (Weighted): {pre:.4f}")
        print(f"Recall (Weighted): {rec:.4f}")
        print(f"TP (YES): {tp} | FN (YES): {fn}")
        print(f"TN (NO): {tn} | FP (NO): {fp}")
        print(f"Unknown Predictions: {num_unknown}")

    # --- 4. 打印最终的 Table 1 ---
    print("\n\n--- [ 最终结果 (复现 Table 1) ] ---")
    
    headers = ["Dataset", "Accuracy", "F1", "Precision", "Recall", "TP", "FN", "TN", "FP", "Unknowns"]
    print(f"{headers[0]:<25} | " + " | ".join(f"{h:<8}" for h in headers[1:]))
    print("-" * 120)
    
    for test_file, metrics in results.items():
        name = test_file.replace("_traj_list.json", "")
        print(f"{name:<25} | "
              f"{metrics['Accuracy'] * 100:>7.2f}% | "
              f"{metrics['F1-Score (Weighted)']:<8.4f} | "
              f"{metrics['Precision (Weighted)']:<8.4f} | "
              f"{metrics['Recall (Weighted)']:<8.4f} | "
              f"{metrics['TP (YES)']:<8} | "
              f"{metrics['FN (YES)']:<8} | "
              f"{metrics['TN (NO)']:<8} | "
              f"{metrics['FP (NO)']:<8} | "
              f"{metrics['Unknowns']:<8}")

# --- [ 4. 启动入口 ] ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SFT model (Table 1 Replication).")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained SFT model directory (e.g., .../SFT_models/round_init/)."
    )
    parser.add_argument(
        "--sft_model_path", 
        type=str, 
        default= "/path/to/checkpoint/saves/easy_r1/experiment_name/SFT_models/round_init/",
        help="Path to the trained SFT model directory (e.g., .../SFT_models/round_init/)."
    )
    parser.add_argument(
        "--test_data_dir", 
        type=str,
        default="/path/to/data/data/test_data",
        help="Directory containing the test JSON files (AITW_ID, etc.)."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/path/to/data/data/test_data",
        help="Directory where the image files (referenced in the JSONs) are stored."
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default="gt",
        help="The key in the JSON file that holds the ground truth label."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误：找不到模型路径 {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.test_data_dir):
        print(f"错误：找不到测试数据路径 {args.test_data_dir}")
        sys.exit(1)

    evaluate_model(args)

"""
python verl/trainer/eval_sft.py --model_path /path/to/checkpoint/saves/easy_r1/experiment_name/SFT_models/round_init/
"""
