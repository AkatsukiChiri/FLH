import torch
import argparse
import logging

DEV = torch.device('cuda:0')

supported_datasets = ['wikitext2', 'ptb', 'c4']

def parser_gen():
    parser = argparse.ArgumentParser()
    
    # General Arguments
    parser.add_argument('--model', type=str, default='/home/mashaobo/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, default='wikitext2', choices=supported_datasets,)
    parser.add_argument('--bsz', type=int, default=4)
    
    # Rotation Arguments
    parser.add_argument("--rotation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--groupsize", type=int, default=-1)
    
    # Activation Quantization Arguments
    parser.add_argument("--a_bits", type=int, default=16)
    parser.add_argument("--a_asym", type=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--a_clip_ratio", type=float, default=1.0)
    
    # Weight Quantization Arguments
    parser.add_argument("--w_bits", type=int, default=16)
    parser.add_argument("--w_asym", type=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--w_clip", type=float, default=1.0)
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False)
    
    # KV-Cache Quantization Arguments
    parser.add_argument("--v_bits", type=int, default=16)
    parser.add_argument("--v_asym", type=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--v_clip_ratio", type=float, default=1.0)
    
    parser.add_argument("--k_bits", type=int, default=16)
    parser.add_argument("--k_asym", type=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--k_clip_ratio", type=float, default=1.0)

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true")
    parser.add_argument('--tasks', nargs='+', default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada"])
    parser.add_argument('--lm_eval_batch_size', type=int, default=128)
    parser.add_argument("--distribute", action="store_true")
    
    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None)
    parser.add_argument('--save_qmodel_path', type=str, default=None)
    
    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.lm_eval:
        from lm_eval import tasks
        from lm_eval import utils as lm_eval_utils
        from lm_eval.tasks import initialize_tasks
        initialize_tasks()
        for task in args.tasks:
            if task not in lm_eval_utils.MultiChoice(tasks.ALL_TASKS):
                raise ValueError(f"Invalid task: {task}")
    
    return args

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )