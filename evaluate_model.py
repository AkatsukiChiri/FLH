#!/usr/bin/env python3
"""
Evaluation script for FLH models on various benchmarks.

This script evaluates:
1. Perplexity on WikiText2
2. Accuracy on common NLP benchmarks (optional, requires lm-evaluation-harness)

Usage:
    python evaluate_model.py --model meta-llama/Llama-2-7b-hf --tasks wikitext
    python evaluate_model.py --model meta-llama/Llama-2-7b-hf --tasks all
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from flh.quantized_model.modeling_llama import FLH_FP16LlamaForCausalLM


def load_model_and_tokenizer(model_name_or_path, device="cuda", dtype=torch.float16, attn_implementation="flash_attention_2"):
    """Load model and tokenizer"""
    print(f"Loading model from {model_name_or_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Load base config and create FLH model
    config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Set attention implementation
    config._attn_implementation = attn_implementation
    
    # Note: FLH_FP16LlamaForCausalLM requires flash_attention_2
    if attn_implementation != "flash_attention_2":
        print(f"⚠ Warning: FLH_FP16LlamaForCausalLM requires flash_attention_2, but {attn_implementation} was specified.")
        print(f"  Forcing flash_attention_2...")
        config._attn_implementation = "flash_attention_2"
    
    # Create model with explicit dtype context to ensure Flash Attention 2 compatibility
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    
    model = FLH_FP16LlamaForCausalLM(config)
    
    torch.set_default_dtype(dtype_old)
    
    # Explicitly convert to target dtype and device
    model = model.to(device=device, dtype=dtype)
    
    # Load pretrained weights
    # Note: You may need to adapt this based on your weight loading mechanism
    print("✓ Model and tokenizer loaded successfully")
    
    model.eval()
    
    return model, tokenizer


def load_wikitext2(tokenizer, seq_length=2048, split="test"):
    """Load and tokenize WikiText2 dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print(f"Loading WikiText2 {split} split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Concatenate all texts
    text = "\n\n".join(dataset["text"])
    
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    
    # Create sliding window chunks
    input_ids = encodings.input_ids[0]
    
    # Split into chunks of seq_length
    num_chunks = (len(input_ids) - 1) // seq_length + 1
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * seq_length
        end_idx = min((i + 1) * seq_length, len(input_ids))
        chunk = input_ids[start_idx:end_idx]
        
        # Pad last chunk if necessary
        if len(chunk) < seq_length:
            pad_length = seq_length - len(chunk)
            chunk = F.pad(chunk, (0, pad_length), value=tokenizer.pad_token_id or tokenizer.eos_token_id)
        
        chunks.append(chunk)
    
    print(f"✓ Loaded {len(chunks)} chunks from WikiText2 {split}")
    return torch.stack(chunks)


def calculate_perplexity(model, input_ids, tokenizer, device="cuda", batch_size=1):
    """
    Calculate perplexity on the given input_ids
    
    Args:
        model: The language model
        input_ids: Tensor of shape [num_samples, seq_length]
        tokenizer: The tokenizer (for pad_token_id)
        device: Device to run on
        batch_size: Batch size for evaluation
        
    Returns:
        perplexity: The perplexity score
        avg_loss: Average cross-entropy loss
    """
    import gc
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    num_batches = (len(input_ids) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        with tqdm(total=num_batches, desc="Calculating perplexity") as pbar:
            for i in range(0, len(input_ids), batch_size):
                batch = input_ids[i:i+batch_size].to(device)
                
                # Forward pass - only get logits, don't compute loss twice
                outputs = model(batch, use_cache=False)
                logits = outputs.logits
                
                # Calculate loss manually
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                
                # Flatten for loss calculation
                shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels_flat = shift_labels.view(-1)
                
                # Count non-padding tokens
                valid_mask = shift_labels_flat != tokenizer.pad_token_id
                valid_tokens = valid_mask.sum().item()
                
                if valid_tokens > 0:
                    # Calculate loss only on valid tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token_id)
                    batch_loss = loss_fct(shift_logits_flat, shift_labels_flat)
                    total_loss += batch_loss.item()
                    total_tokens += valid_tokens
                
                # Clean up to save memory
                del outputs, logits, shift_logits, shift_labels, batch
                if i % 10 == 0:  # Periodic cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
                
                pbar.update(1)
                if total_tokens > 0:
                    pbar.set_postfix({"loss": f"{total_loss/total_tokens:.4f}"})
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return perplexity, avg_loss


def evaluate_lm_harness(model, tokenizer, tasks, device="cuda", batch_size=1):
    """
    Evaluate using lm-evaluation-harness
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        tasks: List of tasks to evaluate (e.g., ['lambada', 'hellaswag', 'winogrande'])
        device: Device to run on
        batch_size: Batch size for evaluation
        
    Returns:
        results: Dictionary of results
    """
    try:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("⚠ lm-evaluation-harness not installed. Skipping additional benchmarks.")
        print("  Install with: pip install lm-eval")
        return None
    
    print(f"\nRunning lm-evaluation-harness on tasks: {tasks}")
    print(f"  Batch size: {batch_size}")
    
    try:
        # Ensure model is on the correct device
        model = model.to(device)
        
        # Wrap model for lm_eval - use HFLM wrapper
        print("  Wrapping model with HFLM...")
        hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
        # Match task names from patterns
        print("  Matching task names...")
        task_names = lm_eval_utils.pattern_match(tasks, ALL_TASKS)
        print(f"  Found tasks: {task_names}")
        
        # Run evaluation
        print("  Running evaluation...")
        eval_results = lm_eval.simple_evaluate(
            hflm,
            tasks=task_names,
            batch_size=batch_size
        )
        
        results_dict = eval_results['results']
        
        # Extract metrics
        print("\n  Processing results...")
        metric_vals = {}
        for task, result in results_dict.items():
            # Try to get normalized accuracy first, then regular accuracy
            if 'acc_norm,none' in result:
                metric_vals[task] = round(result['acc_norm,none'], 4)
            elif 'acc,none' in result:
                metric_vals[task] = round(result['acc,none'], 4)
            elif 'acc_norm' in result:
                metric_vals[task] = round(result['acc_norm'], 4)
            elif 'acc' in result:
                metric_vals[task] = round(result['acc'], 4)
            else:
                # Store all metrics if no accuracy found
                metric_vals[task] = {k: round(v, 4) if isinstance(v, float) else v 
                                    for k, v in result.items()}
        
        # Calculate average accuracy if applicable
        numeric_metrics = [v for v in metric_vals.values() if isinstance(v, (int, float))]
        if numeric_metrics:
            metric_vals['acc_avg'] = round(sum(numeric_metrics) / len(numeric_metrics), 4)
        
        print(f"  ✓ Evaluation complete!")
        
        return metric_vals
        
    except Exception as e:
        print(f"  ✗ Error during lm-eval evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate FLH models on various benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="wikitext",
        help="Comma-separated list of tasks: wikitext, lambada, hellaswag, winogrande, arc_easy, arc_challenge, mmlu, or 'all'"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for evaluation (default: 512, use smaller value if OOM)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager", "sdpa"],
        help="Attention implementation (Note: FLH_FP16LlamaForCausalLM requires flash_attention_2)"
    )
    parser.add_argument(
        "--lm-eval-batch-size",
        type=int,
        default=None,
        help="Batch size for lm-eval tasks (default: same as --batch-size)"
    )
    
    args = parser.parse_args()
    
    # Set lm-eval batch size to main batch size if not specified
    if args.lm_eval_batch_size is None:
        args.lm_eval_batch_size = args.batch_size
    
    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Parse tasks
    if args.tasks.lower() == "all":
        tasks = ["wikitext", "lambada", "hellaswag", "winogrande", "arc_easy", "arc_challenge"]
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
    
    print("=" * 80)
    print("FLH Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tasks: {tasks}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Attention: {args.attn_implementation}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model, 
        device=args.device, 
        dtype=dtype, 
        attn_implementation=args.attn_implementation
    )
    
    # Make sure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    # Evaluate on WikiText2
    if "wikitext" in tasks or "wikitext2" in tasks:
        print("\n" + "=" * 80)
        print("Evaluating on WikiText2")
        print("=" * 80)
        
        input_ids = load_wikitext2(tokenizer, seq_length=args.seq_length, split=args.split)
        
        start_time = time.time()
        perplexity, avg_loss = calculate_perplexity(
            model, 
            input_ids,
            tokenizer,
            device=args.device, 
            batch_size=args.batch_size
        )
        eval_time = time.time() - start_time
        
        results["wikitext2"] = {
            "perplexity": perplexity,
            "loss": avg_loss,
            "eval_time_seconds": eval_time,
            "num_samples": len(input_ids),
            "seq_length": args.seq_length,
        }
        
        print(f"\n✓ WikiText2 Results:")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Evaluation time: {eval_time:.2f}s")
    
    # Evaluate on other benchmarks (if requested)
    other_tasks = [t for t in tasks if t not in ["wikitext", "wikitext2"]]
    if other_tasks:
        print("\n" + "=" * 80)
        print("Evaluating on additional benchmarks")
        print("=" * 80)
        
        lm_results = evaluate_lm_harness(
            model, 
            tokenizer, 
            other_tasks, 
            device=args.device,
            batch_size=args.lm_eval_batch_size
        )
        if lm_results:
            results.update(lm_results)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results")
    print("=" * 80)
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    for task_name, task_results in results.items():
        print(f"\n{task_name.upper()}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
