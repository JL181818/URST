import os
import json
import random
import numpy as np
import torch
import transformers
import logging
import time
from termcolor import cprint
import torch.multiprocessing as mp
import socket
# from paths import *
import yaml

system_msg = """You are an expert in evaluating the performance of an android navigation agent. The agent is designed to help a human user navigate the device to complete a task. Given the user's intent, the agent's action history, and the last two states of the screen, your goal is to decide whether the agent has successfully completed the task or not.

*Output Format*
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>reasoning process here</think><answer>answer here YES or NO</answer>"""


# def set_the_seed(seed):
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     transformers.set_seed(seed)

def load_json(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)

# def save_json(data, data_path):
#     with open(data_path, 'w') as f:
#         json.dump(data, f, indent=4)

# def prepare(args):
#     mp.set_start_method('spawn')
#     set_the_seed(42)
    
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     args.timestamp = timestamp
#     args.output_dir = os.path.join(args.save_dir, timestamp)
#     args.log_dir = os.path.join(args.output_dir, "logs")
#     os.makedirs(args.save_dir, exist_ok=True)
#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(args.log_dir, exist_ok=True)
#     log_file = os.path.join(args.log_dir, f"train_log.txt")
#     logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    
#     def is_port_in_use(port):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             return s.connect_ex(('localhost', port)) == 0
#     def find_available_port(start_port):
#         port = start_port
#         while is_port_in_use(port):
#             port += 1
#         return port
    
#     available_port = find_available_port(args.port)
#     if available_port != args.port:
#         print(f"Port {args.port} is in use. Using port {available_port} instead.")
#     # renew the port
#     args.port = available_port
    
#     logging.info('='*20+'args'+'='*20)
#     for key, value in vars(args).items():
#         logging.info(f"{key}: {value}")
#     logging.info('='*44)

# def get_subdirectory_paths(directory):
#     subdirectories = []
#     for entry in os.listdir(directory):
#         full_path = os.path.join(directory, entry)
#         if os.path.isdir(full_path):
#             subdirectories.append(os.path.abspath(full_path))
#     return subdirectories

# def calculate_entropy(logits):
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)

# def convert2training_data(args, all_data, selected_ids, round_num, mode='sft'):

def convert2training_data(all_data, selected_ids,sft_config_path,  mode='sft'):
    with open(sft_config_path, 'r') as file:
        config = yaml.safe_load(file)
    output_dir = config.get("output_dir", "./sft_output")
    os.makedirs(output_dir, exist_ok=True)
    training_data = []
    selected_data = {selected_id: all_data[selected_id] for selected_id in selected_ids}
    selected_data_path = os.path.join(output_dir, f"train_data_dict_round_init.json")
    with open(selected_data_path, 'w') as f:
        json.dump(selected_data, f, indent=4)
    train_img_dir = config.get("train_img_dir", "./images")
    for selected_id in selected_ids:
        current_data = all_data[selected_id]
        user_intent = current_data["User Intent"]
        action_history = current_data["Action History"]
        user_content = f"User Intent: {user_intent}\n\nAction History: {action_history}\n\nthe second last state and the last state of the screen are shown in the images."
        if len(current_data["image_paths"]) >= 2:
            user_content += "\n<image><image>"
            image_paths = [os.path.join(train_img_dir, current_data["image_paths"][-2]), os.path.join(train_img_dir, current_data["image_paths"][-1])]
        elif len(current_data["image_paths"]) == 1:
            user_content += "\n<image>"
            image_paths = [os.path.join(train_img_dir, current_data["image_paths"][-1])]
            # cprint(("data only have one image:", selected_id), "red")
        else:
            cprint(("data have no image:", selected_id), "red")
        
        thought = current_data['Thoughts']
        status = current_data['gt']
        assistant_content = f'<think>{thought}</think>\n<answer>{status}</answer>'    
        
        if mode == 'sft':
            training_data.append({
                "system": system_msg,
                "messages": [
                    {
                        "content": user_content,
                        "role": "user"
                    },
                    {
                        "content": assistant_content,
                        "role": "assistant"
                    }
                ],
                "images": image_paths,
            })
            
        elif mode == 'st':
            training_data.append({
                "system": system_msg,
                "messages": [
                    {
                        "content": user_content,
                        "role": "user"
                    },
                ],
                "images": image_paths,
                "solution": gt
            })
        else:
            raise ValueError("Invalid mode. Choose either 'sft' or 'st'.")    
            
    return training_data

def set_the_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)

# def plot_hist(data, save_path):
#     import matplotlib.pyplot as plt
#     plt.hist(data, edgecolor='black')
#     plt.xlabel('Entropy')
#     plt.ylabel('Frequency')
#     plt.savefig(save_path)
#     plt.close()