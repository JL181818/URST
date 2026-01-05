import argparse
import subprocess
from utils import convert2training_data
from utils import load_json,set_the_seed
import os
import json
import random
import yaml
from termcolor import cprint
from paths import * # 确保这里导入了 LLAMAF_DIR 和 PROJECT_ROOT
import torch
import sys

def main(args):
    set_the_seed(42)  # 设置随机种子，确保可复现性
    # 微调阶段
    all_data_list = load_json(args.train_files)
    all_data = {item['id']: item for item in all_data_list}
    
    sft_id_dir = args.output_dir
    os.makedirs(sft_id_dir, exist_ok=True)
    

    yes_samples = {k: v for k, v in all_data.items() if v['gt'].lower() == 'yes'}
    no_samples = {k: v for k, v in all_data.items() if v['gt'].lower() == 'no'}
    assert len(yes_samples) >= 150, "Not enough 'YES' samples"
    assert len(no_samples) >= 150, "Not enough 'NO' samples"
    init_yes_ids = random.sample(list(yes_samples.keys()), 150)
    init_no_ids = random.sample(list(no_samples.keys()), 150)
    init_data = {**{k: yes_samples[k] for k in init_yes_ids}, **{k: no_samples[k] for k in init_no_ids}}
    
    selected_ids = list(init_data.keys())

    sft_id_path = os.path.join(sft_id_dir, "selected_id.json")
    with open(sft_id_path, "w") as f:
        json.dump(selected_ids, f, indent=4)

    random.shuffle(selected_ids)
    train_data = convert2training_data(all_data, selected_ids, args.sft_config_path, mode='sft')
    
    # 检查 train_model_sft 的返回值，以避免“假成功”日志
    cprint(f"[INFO] 准备启动 SFT 训练...", "cyan")
    model_output_dir = train_model_sft(
        args.sft_config_path, 
        train_data,
        project_name=args.project_name,
        experiment_name=args.experiment_name
    )
    
    if model_output_dir:
        cprint(f"SFT 训练成功完成。模型保存在: {model_output_dir}", "green")
        print("Finished SFT training for initialization.")
    else:
        cprint(f"SFT 训练失败。请检查上面的 [ERROR] 日志。", "red")
        sys.exit(1) # 以错误码退出


def train_model_sft(sft_config_path, train_data, project_name="easy_r1", experiment_name="qwen2_5_vl_3b_urst_sgpo"):
    cprint(f"--- 启动 SFT 训练流程 ---", "cyan")
    
    # 1. 加载配置
    try:
        with open(sft_config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        cprint(f"[ERROR] 无法加载 SFT 配置文件: {sft_config_path}. 错误: {e}", "red")
        return None

    # 2. 设定路径
    save_checkpoint_path = os.path.join("/path/to/checkpoint/saves", project_name, experiment_name)
    sft_models_dir = os.path.join(save_checkpoint_path, "SFT_models")
    
    # 期望的模型输出目录
    model_output_dir = os.path.join(sft_models_dir, "round_init")
    
    # 训练数据的存放路径
    train_data_path = os.path.join(sft_models_dir, "train_data_sft.json")
    
    os.makedirs(model_output_dir, exist_ok=True)
    cprint(f"[DEBUG] 训练数据将保存至: {train_data_path}", "yellow")
    cprint(f"[DEBUG] 期望的模型输出目录: {model_output_dir}", "yellow")

    # 3. 动态修改配置中的 output_dir
    cprint(f"[DEBUG] YAML 中原始 'output_dir': {config.get('output_dir', '未设置')}", "magenta")
    config['output_dir'] = os.path.abspath(model_output_dir) # 确保使用绝对路径
    config['dataset'] = 'evaluator_sft_data' # 确保使用了正确的数据集
    
    try:
        with open(sft_config_path, 'w') as file:
            yaml.safe_dump(config, file)
        cprint(f"[INFO] 已将更新后的 'output_dir' ({config['output_dir']}) 写回 {sft_config_path}", "green")
    except Exception as e:
        cprint(f"[ERROR] 无法回写 SFT 配置文件: {sft_config_path}. 错误: {e}", "red")
        return None

    # 4. 准备训练数据
    try:
        with open(train_data_path, 'w') as f:
            json.dump(train_data, f, indent=4)
    except Exception as e:
        cprint(f"[ERROR] 写入训练数据失败: {train_data_path}. 错误: {e}", "red")
        return None
        
    # 5. 修改 LLaMA-Factory 的 dataset_info.json
    try:
        dataset_info_path = f'{LLAMAF_DIR}/data/dataset_info.json'
        with open(dataset_info_path, 'r') as file:
            dataset_info = json.load(file)
        
        dataset_info['evaluator_sft_data']["file_name"] = os.path.abspath(train_data_path)
        
        with open(dataset_info_path, 'w') as file:
            json.dump(dataset_info, file, indent=4)
        cprint(f"[INFO] 已更新 {dataset_info_path} 指向新的训练数据。", "green")
    except Exception as e:
        cprint(f"[ERROR] 修改 'dataset_info.json' 失败. 路径: {dataset_info_path}. 错误: {e}", "red")
        return None

    # 6. 设置环境变量
    devices_str = get_available_devices([0,2,3,4,5,7])
    if not devices_str:
        cprint("[ERROR] 没有可用的 GPU，SFT 训练终止。", "red")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices_str
    env['NCCL_P2P_LEVEL'] = 'NVL'
    env["FORCE_TORCHRUN"] = "1"
    
    conda_prefix = f'source /root/anaconda3/etc/profile.d/conda.sh && conda activate sft && '
    cmd = f"""{conda_prefix} \
        DISABLE_VERSION_CHECK=1 \
        CUDA_VISIBLE_DEVICES={devices_str} \
        llamafactory-cli train {sft_config_path}"""

    # 替换为健壮的子进程调用，包含实时日志和错误检查
    try:
        cprint('--------------------', "yellow")
        cprint(f"[INFO] 准备执行 SFT 命令:", "cyan")
        print(cmd) # 打印完整的多行命令
        cprint('--------------------', "yellow")
        cprint(f"[INFO] LLaMA-Factory 工作目录 (cwd): {LLAMAF_DIR}", "cyan")
        cprint(f"[INFO] 环境变量 CUDA_VISIBLE_DEVICES: {devices_str}", "cyan")
        cprint(f"[INFO] SFT 训练开始... 将实时打印 STDOUT/STDERR。", "cyan")
        
        handler = subprocess.Popen(
            cmd, 
            cwd=LLAMAF_DIR, 
            shell=True, 
            env=env, 
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # 行缓冲
        )

        # 实时打印 stdout
        if handler.stdout:
            for line in iter(handler.stdout.readline, ''):
                print(f"[LLaMA-Factory STDOUT] {line.strip()}", flush=True)

        # 等待进程结束并获取 stderr
        stdout_data, stderr_data = handler.communicate()

        # 打印剩余的 stdout (如果有的话)
        if stdout_data:
             print(f"[LLaMA-Factory STDOUT-FINAL] {stdout_data.strip()}", flush=True)

        # 检查返回码
        if handler.returncode == 0:
            cprint(f"[SUCCESS] SFT 训练进程成功结束 (Code 0)。", "green")
            
            # 最终验证
            if os.path.exists(model_output_dir) and len(os.listdir(model_output_dir)) > 0:
                cprint(f"[VERIFIED] 模型已成功保存到: {model_output_dir}", "green")
                cprint(f"[VERIFIED] 目录内容: {os.listdir(model_output_dir)}", "green")
                return model_output_dir # 成功，返回路径
            else:
                cprint(f"[WARNING] 进程返回 0，但在 {model_output_dir} 中未找到模型文件！", "red")
                cprint(f"[WARNING] 请检查 {sft_config_path} 中的 'output_dir' 和 'save_strategy'。", "red")
                return None # 成功但没文件，返回 None
        else:
            cprint(f"[ERROR] SFT 训练进程失败 (Code {handler.returncode})。", "red")
            cprint(f"--- [LLaMA-Factory STDERR] ---", "red")
            print(stderr_data)
            cprint(f"--- [LLaMA-Factory STDERR END] ---", "red")
            return None # 失败，返回 None

    except Exception as e:
        cprint(f"[FATAL ERROR] SFT 子进程执行失败: {e}", "red")
        if 'handler' in locals() and handler:
            handler.kill() # 确保子进程被杀死
        return None


def get_available_devices(devices):
    available_devices = []
    for device in devices:
        try:
            torch.cuda.set_device(device)
            torch.cuda.current_device() # 额外测试
            available_devices.append(str(device))
            cprint(f"[DEBUG] GPU {device} is available.", "green")
        except RuntimeError as e:
            cprint(f"[DEBUG] GPU {device} is in use or unavailable: {e}", "yellow")
    
    if not available_devices:
        cprint(f"[ERROR] No available GPUs found in the provided list: {devices}", "red")
        return ""
        
    return ",".join(available_devices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_config_path", type=str, default=f"{LLAMAF_DIR}/examples/train_full/qwen2_5_vl_evaluator_full_sft.yaml", help="Path to sft config file")
    parser.add_argument("--train_files", type=str, default=f'/path/to/data/train_data/processed_candidate_data_labelled_list.json', help='Path to training data files')
    parser.add_argument("--project_name", type=str, default="easy_r1")
    parser.add_argument("--experiment_name", type=str, default="experiment_name")  
    args = parser.parse_args()
    
    cprint("--- 启动参数 ---", "yellow")
    cprint(args, "green")
    cprint("------------------", "yellow")
    
    main(args)

# python ./EasyR1/verl/trainer/sft.py
# 修改 EasyR1/LLaMA-Factory/examples/train_full/qwen2_5_vl_evaluator_full_sft.yaml