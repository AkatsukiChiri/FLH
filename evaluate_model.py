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
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from flh.quantized_model.modeling_llama import FLH_FP16LlamaForCausalLM, FLH_LlamaForCausalLM


def load_original_model(model_name_or_path, device="cuda", dtype=torch.float16, attn_implementation="flash_attention_2"):
    """Load original LlamaForCausalLM model"""
    print(f"Loading original LlamaForCausalLM from {model_name_or_path}...")
    from transformers import LlamaForCausalLM
    
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
        device_map=device
    )
    
    torch.set_default_dtype(dtype_old)
    model = model.half().to(device=device)
    model.eval()
    
    print("✓ Original model loaded successfully")
    return model


def load_model_and_tokenizer(model_name_or_path, device="cuda", dtype=torch.float16, attn_implementation="flash_attention_2", use_quantized=False, load_quantized_path=None, save_quantized_path=None, weight_sym=False, act_sym=True):
    """Load model and tokenizer
    
    Args:
        model_name_or_path: Path to original model
        device: Target device
        dtype: Data type
        attn_implementation: Attention implementation
        use_quantized: Whether to use quantized model
        load_quantized_path: Path to pre-quantized model (fast loading)
        save_quantized_path: Path to save quantized model after quantization
    """
    # 如果指定了加载路径，直接加载已量化的模型（快速）
    if load_quantized_path:
        print(f"Loading pre-quantized model from {load_quantized_path}...")
        model = FLH_LlamaForCausalLM.load_quantized(load_quantized_path, target_device=device)
        tokenizer = AutoTokenizer.from_pretrained(load_quantized_path)
        model.eval()
        print("✓ Pre-quantized model loaded successfully (fast!)")
        return model, tokenizer
    
    # 否则，从原始模型加载
    print(f"Loading model from {model_name_or_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Note: FLH_FP16LlamaForCausalLM requires flash_attention_2
    if attn_implementation != "flash_attention_2":
        print(f"⚠ Warning: FLH_FP16LlamaForCausalLM requires flash_attention_2, but {attn_implementation} was specified.")
        print(f"  Forcing flash_attention_2...")
        attn_implementation = "flash_attention_2"
    
    # Load original Llama model on CPU to avoid OOM
    print("  Loading original LlamaForCausalLM on CPU (to avoid OOM)...")
    from transformers import LlamaForCausalLM
    
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    
    original_model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
        device_map="cpu"  # Load on CPU first
    )
    
    torch.set_default_dtype(dtype_old)
    
    # Convert to FLH model using from_float (conversion happens on CPU)
    if use_quantized:
        print("  Converting to FLH_LlamaForCausalLM (quantized) using from_float...")
        model = FLH_LlamaForCausalLM.from_float(
            original_model, 
            target_device=device,
            weight_sym=weight_sym,
            act_sym=act_sym,
            save_quantized_path=save_quantized_path  # 量化后立即保存（在移到GPU之前）
        )
        
        # 如果保存了模型，也保存tokenizer
        if save_quantized_path:
            print(f"  Saving tokenizer to {save_quantized_path}...")
            tokenizer.save_pretrained(save_quantized_path)
    else:
        print("  Converting to FLH_FP16LlamaForCausalLM using from_float...")
        model = FLH_FP16LlamaForCausalLM.from_float(original_model, target_device=device)
    
    print("✓ Model and tokenizer loaded successfully")
    
    model.eval()
    
    # Clean up original model to save memory
    print("  Cleaning up original model from CPU memory...")
    del original_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return model, tokenizer


def load_wikitext2(tokenizer, seq_length=2048, split="test"):
    """
    Load and tokenize WikiText2 dataset following GPTQ evaluation method.
    
    Returns tokenized data ready for evaluation.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print(f"Loading WikiText2 {split} split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Concatenate all text
    text = "\n\n".join(dataset["text"])
    
    # Tokenize - following GPTQ: use return_tensors='pt'
    print(f"  Tokenizing text...")
    testenc = tokenizer(text, return_tensors='pt')
    
    print(f"✓ Loaded WikiText2 {split}")
    print(f"  Total tokens: {testenc.input_ids.numel()}")
    print(f"  Sequence length: {seq_length}")
    
    return testenc


def calculate_perplexity(model, testenc, device="cuda", seq_length=2048):
    """
    Calculate perplexity following GPTQ evaluation method.
    
    This implementation exactly follows the llama_eval function from GPTQ:
    - Split data into fixed-size chunks of seq_length
    - Calculate NLL for each chunk
    - Sum all NLLs and compute perplexity
    
    Args:
        model: The language model
        testenc: Tokenized test data (from tokenizer with return_tensors='pt')
        device: Device to run on
        seq_length: Sequence length for evaluation (default: 2048)
        
    Returns:
        perplexity: Token-level perplexity score
    """
    print('Evaluating perplexity...')
    
    model.eval()
    
    # Following GPTQ: testenc.input_ids
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seq_length
    
    print(f"  Number of samples: {nsamples}")
    print(f"  Sequence length: {seq_length}")
    
    testenc = testenc.to(device)
    
    nlls = []
    with torch.no_grad():
        # Following GPTQ evaluation loop
        for i in tqdm(range(nsamples), desc="Calculating perplexity"):
            # Get the current batch
            batch = testenc[:, (i * seq_length):((i + 1) * seq_length)]
            
            # Forward pass
            outputs = model(batch, use_cache=False)
            lm_logits = outputs.logits
            
            # Following GPTQ: calculate loss
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:]
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # Following GPTQ: neg_log_likelihood = loss.float() * seq_length
            neg_log_likelihood = loss.float() * seq_length
            nlls.append(neg_log_likelihood)
            
            # Clean up
            del outputs, lm_logits, shift_logits, shift_labels, batch
            if i % 10 == 0:
                torch.cuda.empty_cache()
    
    # Following GPTQ: ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_length))
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_length))
    
    print(f"  Perplexity: {ppl.item():.4f}")
    
    return ppl.item()


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


def evaluate_model_on_tasks(model, tokenizer, tasks, args, model_name="Model"):
    """
    Evaluate a model on specified tasks
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        tasks: List of tasks to evaluate on
        args: Command-line arguments
        model_name: Name for logging (e.g., "FLH Model", "Original Model")
        
    Returns:
        Dictionary of results
    """
    results = {}
    
    # Evaluate on WikiText2
    if "wikitext" in tasks or "wikitext2" in tasks:
        print("\n" + "=" * 80)
        print(f"Evaluating {model_name} on WikiText2")
        print("=" * 80)
        
        testenc = load_wikitext2(
            tokenizer, 
            seq_length=args.seq_length, 
            split=args.split
        )
        
        start_time = time.time()
        ppl = calculate_perplexity(
            model, 
            testenc,
            device=args.device,
            seq_length=args.seq_length
        )
        eval_time = time.time() - start_time
        
        results["wikitext2"] = {
            "perplexity": ppl,
            "eval_time_seconds": eval_time,
            "total_tokens": testenc.input_ids.numel(),
            "seq_length": args.seq_length,
            "num_samples": testenc.input_ids.numel() // args.seq_length,
        }
        
        print(f"\n✓ {model_name} WikiText2 Results:")
        print(f"  Perplexity: {ppl:.4f}")
        print(f"  Evaluation time: {eval_time:.2f}s")
    
    # Evaluate on other benchmarks (if requested)
    other_tasks = [t for t in tasks if t not in ["wikitext", "wikitext2"]]
    if other_tasks:
        print("\n" + "=" * 80)
        print(f"Evaluating {model_name} on additional benchmarks")
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
    
    return results


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
        default=2048,
        help="Sequence length for WikiText evaluation (default: 2048, following GPTQ)"
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
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use quantized FLH_LlamaForCausalLM instead of FP16 version"
    )
    parser.add_argument(
        "--compare-with-original",
        action="store_true",
        default=True,
        help="Compare FLH model with original model (default: True)"
    )
    parser.add_argument(
        "--no-compare",
        dest="compare_with_original",
        action="store_false",
        help="Skip comparison with original model"
    )
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Skip FLH model and only evaluate original model"
    )
    parser.add_argument(
        "--save-quantized",
        type=str,
        default=None,
        help="Save quantized model to specified directory (e.g., ./quantized_model)"
    )
    parser.add_argument(
        "--load-quantized",
        type=str,
        default=None,
        help="Load pre-quantized model from specified directory (fast, skips quantization)"
    )
    parser.add_argument(
        "--weight-sym",
        action="store_true",
        help="Use symmetric quantization for weights (default: False, asymmetric)"
    )
    parser.add_argument(
        "--act-sym",
        dest="act_sym",
        action="store_true",
        help="Use symmetric quantization for activations (default: True)"
    )
    parser.add_argument(
        "--no-act-sym",
        dest="act_sym",
        action="store_false",
        help="Use asymmetric quantization for activations"
    )
    
    # 设置默认值（在add_argument之后）
    parser.set_defaults(act_sym=True, weight_sym=False)
    
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
    if args.original_only:
        print("Original Model Evaluation (FLH Skipped)")
    elif args.compare_with_original:
        print("FLH Model Evaluation with Original Model Comparison")
    else:
        print("FLH Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    if not args.original_only:
        print(f"FLH Model type: {'FLH_LlamaForCausalLM (Quantized)' if args.quantized else 'FLH_FP16LlamaForCausalLM'}")
    print(f"Evaluate FLH model: {not args.original_only}")
    print(f"Compare with original: {args.compare_with_original or args.original_only}")
    print(f"Tasks: {tasks}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Attention: {args.attn_implementation}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 80)
    
    # Load tokenizer (only once)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize results
    flh_results = None
    original_results = None
    comparison = {}
    
    # ==================== Evaluate FLH Model (if not skipped) ====================
    if not args.original_only:
        print("\n" + "=" * 80)
        print("PART 1: Evaluating FLH Model")
        print("=" * 80)
        
        # 如果指定了加载路径，从预量化模型加载（快速）
        if args.load_quantized:
            print(f"🚀 Fast loading mode: Using pre-quantized model from {args.load_quantized}")
            flh_model, _ = load_model_and_tokenizer(
                args.model, 
                device=args.device, 
                dtype=dtype, 
                attn_implementation=args.attn_implementation,
                use_quantized=args.quantized,
                load_quantized_path=args.load_quantized,
                weight_sym=args.weight_sym,
                act_sym=args.act_sym
            )
        else:
            # 否则从原始模型量化（慢），可选择保存
            flh_model, _ = load_model_and_tokenizer(
                args.model, 
                device=args.device, 
                dtype=dtype, 
                attn_implementation=args.attn_implementation,
                use_quantized=args.quantized,
                save_quantized_path=args.save_quantized,  # 量化完立即保存（在CPU上）
                weight_sym=args.weight_sym,
                act_sym=args.act_sym
            )
        
        # Evaluate FLH model
        flh_results = evaluate_model_on_tasks(
            flh_model, 
            tokenizer, 
            tasks, 
            args, 
            model_name="FLH Model"
        )
        
        comparison["flh_model"] = flh_results
        
        # 注意：如果使用了--save-quantized，模型已经在量化阶段保存了（在CPU上，更快）
    
    # Conditionally evaluate original model
    
    if args.compare_with_original or args.original_only:
        # Unload FLH model to free GPU memory (if FLH was evaluated)
        if not args.original_only and flh_model is not None:
            print("\n" + "=" * 80)
            print("Unloading FLH Model to free GPU memory")
            print("=" * 80)
            del flh_model
            torch.cuda.empty_cache()
            gc.collect()
            print("✓ FLH Model unloaded, GPU memory freed")
        
        # ==================== Evaluate Original Model ====================
        print("\n" + "=" * 80)
        if args.original_only:
            print("Evaluating Original Model")
        else:
            print("PART 2: Evaluating Original Model")
        print("=" * 80)
        
        # Load original model
        original_model = load_original_model(
            args.model,
            device=args.device,
            dtype=dtype,
            attn_implementation=args.attn_implementation
        )
        
        # Evaluate original model
        original_results = evaluate_model_on_tasks(
            original_model, 
            tokenizer, 
            tasks, 
            args, 
            model_name="Original Model"
        )
        
        # Unload original model
        print("\n" + "=" * 80)
        print("Unloading Original Model")
        print("=" * 80)
        del original_model
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Original Model unloaded")
        
        # ==================== Compare Results ====================
        comparison["original_model"] = original_results
        
        # Only compare if both models were evaluated
        if not args.original_only and flh_results is not None:
            print("\n" + "=" * 80)
            print("COMPARISON: FLH Model vs Original Model")
            print("=" * 80)
            
            comparison["differences"] = {}
            
            # Calculate differences for each task
            for task_name in flh_results.keys():
                if task_name in original_results:
                    flh_task = flh_results[task_name]
                    orig_task = original_results[task_name]
                    
                    if isinstance(flh_task, dict) and isinstance(orig_task, dict):
                        comparison["differences"][task_name] = {}
                        for metric in flh_task.keys():
                            if metric in orig_task and isinstance(flh_task[metric], (int, float)) and isinstance(orig_task[metric], (int, float)):
                                diff = flh_task[metric] - orig_task[metric]
                                rel_diff = (diff / orig_task[metric] * 100) if orig_task[metric] != 0 else 0
                                comparison["differences"][task_name][metric] = {
                                    "flh": flh_task[metric],
                                    "original": orig_task[metric],
                                    "absolute_diff": diff,
                                    "relative_diff_percent": rel_diff
                                }
            
            # Print comparison
            for task_name, diffs in comparison["differences"].items():
                print(f"\n{task_name.upper()}:")
                for metric, values in diffs.items():
                    print(f"  {metric}:")
                    print(f"    FLH Model:      {values['flh']:.4f}")
                    print(f"    Original Model: {values['original']:.4f}")
                    print(f"    Absolute Diff:  {values['absolute_diff']:+.4f}")
                    print(f"    Relative Diff:  {values['relative_diff_percent']:+.2f}%")
    else:
        # If not comparing, clean up FLH model if it exists
        if not args.original_only and 'flh_model' in locals():
            del flh_model
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results")
    print("=" * 80)
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    if flh_results is not None:
        print("\n--- FLH MODEL ---")
        for task_name, task_results in flh_results.items():
            print(f"\n{task_name.upper()}:")
            if isinstance(task_results, dict):
                for metric, value in task_results.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            else:
                print(f"  {task_results}")
    
    if original_results is not None:
        print("\n--- ORIGINAL MODEL ---")
        for task_name, task_results in original_results.items():
            print(f"\n{task_name.upper()}:")
            if isinstance(task_results, dict):
                for metric, value in task_results.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            else:
                print(f"  {task_results}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
