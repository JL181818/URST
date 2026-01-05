from ..single_controller.ray import  RayWorkerGroup
from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from omegaconf import OmegaConf
from .config import PPOConfig
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import json
import os
import re
import argparse
from torchdata.stateful_dataloader import StatefulDataLoader
from ..utils.dataset import collate_fn, RLHFDataset
from ..utils.tokenizer import get_processor, get_tokenizer
from .ray_trainer import ResourcePoolManager, Role
from ..single_controller.ray.base import create_colocated_worker_cls
from ..workers.fsdp_workers import FSDPWorker
import ray
from ..single_controller.ray import RayClassWithInitArgs
def parse_answer_from_output(model_output: str) -> str:
    """
    Parses "YES" or "NO" from the model output.
    """
    match = re.search(r'<answer>(.*?)</answer>', model_output, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip().upper()
        if "YES" in answer:
            return "YES"
        if "NO" in answer:
            return "NO"
    
    cleaned_output = model_output.strip().upper()
    if cleaned_output == "YES":
        return "YES"
    if cleaned_output == "NO":
        return "NO"
            
    return "UNKNOWN"

def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--config_path", type=str, default="/urst_project/EasyR1/examples/config.yaml", help="Path to the configuration file")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer/processor")
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True, help="List of checkpoint paths to evaluate")
    parser.add_argument("--checkpoint_names", type=str, nargs='+', help="List of names for the checkpoints")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Base directory for test data and images")
    parser.add_argument("--test_files", type=str, nargs='+', required=True, help="List of test files (JSON)")
    parser.add_argument("--format_prompt_path", type=str, default="/urst_project/EasyR1/examples/format_prompt/urst.jinja", help="Path to the format prompt jinja file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    
    args = parser.parse_args()

    # 1. Define the models to evaluate
    if args.checkpoint_names:
        if len(args.checkpoint_names) != len(args.checkpoints):
            raise ValueError("Number of checkpoint names must match number of checkpoints")
        models_to_evaluate = list(zip(args.checkpoint_names, args.checkpoints))
    else:
        models_to_evaluate = [(f"round_{i}", ckpt) for i, ckpt in enumerate(args.checkpoints)]

    # 2. Setup Ray and Worker Groups (Done once)
    default_config = OmegaConf.structured(PPOConfig())


    config_path = args.config_path
    file_config = OmegaConf.load(config_path)
    ppo_config = OmegaConf.merge(default_config, file_config)

    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()
    # Instantiate tokenizer/processor once (assuming architecture doesn't change between rounds)
    tokenizer = get_tokenizer(
        args.tokenizer_path,
        override_chat_template=None,
        trust_remote_code=False,
        use_fast=True,
    )
    processor = get_processor(
        args.tokenizer_path,
        override_chat_template=None,
        trust_remote_code=False,
        use_fast=True,
    )
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
    }

    global_pool_id = "global_pool"
    mapping = {
        Role.ActorRolloutRef: global_pool_id,
        Role.Critic: global_pool_id,
    }
        
    all_wg: dict[str, FSDPWorker] = {}

    runtime_env_vars = {}
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices:
        runtime_env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"[Final Evaluation] CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    else:
        print("[Final Evaluation] CUDA_VISIBLE_DEVICES not set; Ray will use all visible GPUs.")

    nccl_p2p = os.environ.get("NCCL_P2P_LEVEL", "").strip()
    if nccl_p2p:
        runtime_env_vars["NCCL_P2P_LEVEL"] = nccl_p2p

    ray_init_kwargs = {}
    if runtime_env_vars:
        ray_init_kwargs["runtime_env"] = {"env_vars": runtime_env_vars}

    if cuda_visible_devices:
        num_visible_gpus = len([gpu for gpu in cuda_visible_devices.split(",") if gpu.strip()])
        if num_visible_gpus > 0:
            ray_init_kwargs["num_gpus"] = num_visible_gpus

    ray.init(**ray_init_kwargs)
    available_gpus = int(ray.available_resources().get("GPU", 0))
    if available_gpus <= 0:
        raise RuntimeError("[Final Evaluation] No GPUs detected by Ray. Check CUDA visibility before running tests.")

    trainer_cfg = getattr(ppo_config, "trainer", None)
    desired_nodes = max(1, getattr(trainer_cfg, "nnodes", 1))
    desired_gpus_per_node = max(1, getattr(trainer_cfg, "n_gpus_per_node", available_gpus or 1))
    desired_total_gpus = desired_nodes * desired_gpus_per_node

    if available_gpus < desired_total_gpus:
        print(
            f"[Final Evaluation] WARNING: Requested {desired_total_gpus} GPUs "
            f"({desired_nodes} nodes x {desired_gpus_per_node} GPUs) but only {available_gpus} are available. "
            "Falling back to the available GPUs."
        )
        effective_nodes = min(desired_nodes, available_gpus)
        base = available_gpus // effective_nodes
        extra = available_gpus % effective_nodes
        per_node_counts = [base + (1 if idx < extra else 0) for idx in range(effective_nodes)]
    else:
        effective_nodes = desired_nodes
        per_node_counts = [desired_gpus_per_node] * desired_nodes

    resource_pool_spec = {
        global_pool_id: per_node_counts,
    }
    print(f"[Final Evaluation] Using resource pool spec per node: {per_node_counts}")

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}
    
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
    
    
    actor_rollout_ref_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRolloutRef], config=ppo_config.worker, role="actor_rollout_ref"
    )
    resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
    

    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
    actor_rollout_ref_wg.init_model()
    worker_visible_envs = actor_rollout_ref_wg.execute_all_sync("get_cuda_visible_devices")
    print(f"[Final Evaluation] Worker CUDA_VISIBLE_DEVICES settings: {worker_visible_envs}")

    # Prepare common paths
    base_test_dir = args.test_data_dir
    image_dir = args.test_data_dir
    answer_key = "gt"


    test_files = args.test_files

    

    rollout_engine_ready = False

    # =========================================================================
    # MAIN LOOP: Iterate over each model checkpoint
    # =========================================================================
    for round_name, model_path in models_to_evaluate:
        print(f"\n{'='*50}")
        print(f"STARTING EVALUATION FOR: {round_name}")
        print(f"Path: {model_path}")
        print(f"{'='*50}\n")

        # Load the specific checkpoint
        try:
            actor_rollout_ref_wg.load_checkpoint(model_path)
        except Exception as e:
            print(f"[Final Evaluation] ERROR loading checkpoint for {round_name}: {e}")
            continue

        # Ensure rollout engine reflects the freshly loaded weights
        if rollout_engine_ready:
            actor_rollout_ref_wg.release_rollout_engine()
            rollout_engine_ready = False

        try:
            actor_rollout_ref_wg.prepare_rollout_engine()
            rollout_engine_ready = True
        except Exception as e:
            print(f"[Final Evaluation] ERROR preparing rollout engine for {round_name}: {e}")
            continue

        # Reset metrics containers for this specific model
        all_results = {}
        all_acc = []
        all_f1 = []

        # Inner Loop: Iterate over test files
        for file_name in test_files:
            file_path = os.path.join(base_test_dir, file_name)
            print(f"[Final Evaluation] Processing file: {file_path}")
            if not os.path.exists(file_path):
                print(f"[Final Evaluation] WARNING: Test file not found, skipping: {file_path}")
                continue

            print(f"[Final Evaluation] [{round_name}] Evaluating: {file_name}")

            try:
                test_dataset = RLHFDataset(
                    data_path=file_path,
                    tokenizer=tokenizer,
                    processor=processor,
                    prompt_key="prompt",
                    answer_key=answer_key,
                    image_key="image_paths",
                    video_key="videos",
                    image_dir=image_dir,
                    video_fps=2.0,
                    max_prompt_length=7168,
                    truncation="right",
                    format_prompt=args.format_prompt_path,
                    min_pixels=262144,
                    max_pixels=802816,
                    filter_overlong_prompts=False,
                )
                
                eval_batch_size = min(8, len(test_dataset))
                test_dataloader = StatefulDataLoader(
                    dataset=test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=False,
                )
                
                if len(test_dataloader) == 0:
                    print(f"[Final Evaluation] WARNING: DataLoader for {file_name} is empty. Skipping.")
                    continue

            except Exception as e:
                print(f"[Final Evaluation] ERROR: Failed to create DataLoader for {file_name}. {e}")
                import traceback
                traceback.print_exc()
                continue

            predictions = []
            ground_truths = []
            raw_model_outputs = [] # needed for unknown logging

            # Inference Loop
            for batch_idx, batch_dict in enumerate(tqdm(test_dataloader, desc=f"Testing {file_name} ({round_name})")):
                try:
                    batch_ground_truths = batch_dict.get("ground_truth", None)
                    
                    if batch_ground_truths is None:
                        continue
                    
                    if isinstance(batch_ground_truths, np.ndarray):
                        batch_ground_truths = batch_ground_truths.tolist()
                    elif not isinstance(batch_ground_truths, list):
                        batch_ground_truths = [batch_ground_truths]
                    
                    test_batch = DataProto.from_single_dict(batch_dict)
                    
                    gen_batch = test_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                    
                    gen_batch.meta_info = {"temperature": 1.0,
                                            "top_p": 0.9,
                                            "n": 1}
                    gen_batch.meta_info["do_sample"] = False
                    gen_batch.meta_info["min_pixels"] = 262144
                    gen_batch.meta_info["max_pixels"] = 802816
                    gen_batch.meta_info["video_fps"] = 2.0
                    
                    gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, actor_rollout_ref_wg.world_size)
                    test_output_gen_batch = actor_rollout_ref_wg.generate_sequences(gen_batch)
                    test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

                    output_ids = test_output_gen_batch.batch["responses"]
                    batch_raw_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    parsed_answers = [parse_answer_from_output(raw_output) for raw_output in batch_raw_outputs]
                    
                    predictions.extend(parsed_answers)
                    ground_truths.extend(batch_ground_truths)
                    raw_model_outputs.extend(batch_raw_outputs)
                
                except Exception as e:
                    print(f"[Final Evaluation] ERROR during batch {batch_idx} inference: {e}")
                    import traceback
                    traceback.print_exc()

            # Metric Calculation
            if not ground_truths:
                print(f"[Final Evaluation] ERROR: No ground truths collected for {file_name} in {round_name}.")
                continue

            ground_truths = [str(gt).upper() for gt in ground_truths]
            predictions = [str(pred).upper() for pred in predictions]
            
            valid_indices = [i for i, (gt, pred) in enumerate(zip(ground_truths, predictions)) 
                            if gt in ["YES", "NO"] and pred in ["YES", "NO"]]
            
            # Log Unknowns
            unknown_indices = [i for i, (gt, pred) in enumerate(zip(ground_truths, predictions)) 
                            if gt in ["YES", "NO"] and pred == "UNKNOWN"]
            
            if unknown_indices:
                os.makedirs(args.output_dir, exist_ok=True)
                unknown_log_path = os.path.join(
                    args.output_dir, 
                    f"unknown_preds_{round_name}_{file_name.replace('.json', '')}.json"
                )
                unknown_samples = []
                for idx in unknown_indices:
                    unknown_samples.append({
                        "index": idx,
                        "ground_truth": ground_truths[idx],
                        "prediction": predictions[idx],
                        "raw_output": raw_model_outputs[idx] if idx < len(raw_model_outputs) else "N/A"
                    })
                try:
                    with open(unknown_log_path, "w") as f:
                        json.dump(unknown_samples, f, indent=4, ensure_ascii=False)
                except Exception:
                    pass

            if not valid_indices:
                print(f"[Final Evaluation] WARNING: No valid predictions for {file_name} ({round_name}).")
                
            filtered_gts = [ground_truths[i] for i in valid_indices]
            filtered_preds = [predictions[i] for i in valid_indices]
            
            acc = accuracy_score(filtered_gts, filtered_preds) if filtered_gts else 0.0
            f1 = f1_score(filtered_gts, filtered_preds, average="weighted", labels=["YES", "NO"]) if filtered_gts else 0.0
            
            all_results[file_name] = {
                "accuracy": acc,
                "f1_score_weighted": f1,
                "total_samples": len(ground_truths),
                "valid_samples": len(filtered_gts),
                "unknown_predictions": len(predictions) - len(valid_indices)
            }
            all_acc.append(acc)
            all_f1.append(f1)
            
            print(f"[Final Evaluation] Results for {file_name} ({round_name}): Acc={acc:.4f}, F1={f1:.4f}")

        # Average results for this round
        if all_acc:
            avg_acc = np.mean(all_acc)
            avg_f1 = np.mean(all_f1)
            all_results["average"] = {
                "accuracy_avg": avg_acc,
                "f1_score_weighted_avg": avg_f1
            }
            print(f"\n[Final Evaluation] {round_name} Average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}")

        # Save results for this round
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"final_evaluation_results_{round_name}.json")
        try:
            with open(save_path, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f"[Final Evaluation] Saved {round_name} results to: {save_path}")
        except Exception as e:
            print(f"[Final Evaluation] ERROR: Failed to save results for {round_name}. {e}")

    # Cleanup
    if rollout_engine_ready:
        actor_rollout_ref_wg.release_rollout_engine()

if __name__ == "__main__":
    main()
