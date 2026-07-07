"""
    python experiments/text_recovery.py --config configs/default.yaml \
        --dataset-name coco_two --task-type spatial_relative \
        --question-format open --n-samples 99999 \
        --results-subdir qwen/text_recovery
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vlm_spatial.config import load_config
from vlm_spatial.data import create_question, find_token_ranges, load_dataset
from vlm_spatial.hooks import get_language_layers
from vlm_spatial.model import load_model


def prepare_inputs(processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )


def collect_hidden_states(model, inputs, layers):
    hidden_states = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 2:
                hs = hs.unsqueeze(0)
            hidden_states[layer_idx] = hs.detach().clone()
        return hook_fn

    lm_layers = get_language_layers(model)
    for layer_idx in layers:
        handle = lm_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    logits = outputs.logits[0, -1, :]
    return hidden_states, logits


def run_with_text_patch(model, inputs, clean_states, text_indices, layers,
                        source_text_indices=None):
    """Run forward pass replacing text hidden states from clean run at ALL layers."""
    if source_text_indices is None:
        source_text_indices = text_indices
    hooks = []

    def make_patch_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 2:
                hs = hs.unsqueeze(0)
            patched = hs.clone()
            clean = clean_states[layer_idx]
            for src_idx, tgt_idx in zip(source_text_indices, text_indices):
                patched[0, tgt_idx, :] = clean[0, src_idx, :]
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched
        return hook_fn

    lm_layers = get_language_layers(model)
    for layer_idx in layers:
        handle = lm_layers[layer_idx].register_forward_hook(make_patch_hook(layer_idx))
        hooks.append(handle)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    logits = outputs.logits[0, -1, :]
    return logits


def main():
    parser = argparse.ArgumentParser(
        description="Text recovery under image corruption"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--task-type", type=str, default="spatial_relative",
                        choices=["spatial_relative", "spatial_absolute", "recognition"])
    parser.add_argument("--question-format", type=str, default="open",
                        choices=["forced_choice", "open"])
    parser.add_argument("--corruption", type=str, default="noise",
                        choices=["black", "white", "noise"])
    parser.add_argument("--n-samples", type=int, default=99999)
    parser.add_argument("--question-override", type=str, default=None,
                        help="Override the question text (e.g., 'What is the shape of the object in the image?'). "
                             "Format suffix (Answer with...) is added automatically.")
    parser.add_argument("--n-options", type=int, default=None,
                        help="For forced_choice: randomly select N-1 distractors + GT per sample. "
                             "If None, uses all valid answers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-subdir", type=str, default="text_recovery")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model, processor = load_model(config["model_path"])
    dataset, prepositions = load_dataset(
        config["dataset_path"], args.dataset_name, task_type=args.task_type
    )

    n_layers = len(get_language_layers(model))
    all_layers = list(range(n_layers))
    n_samples = min(args.n_samples, len(dataset))

    # Build first-token IDs for constrained eval
    answer_token_ids = {}
    for prep in prepositions:
        token_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
        answer_token_ids[prep] = token_ids[0]

    counts = {"recovered": 0, "not_recovered": 0, "skipped": 0}
    results_per_sample = []

    # Pre-compute per-sample options for n-options mode
    all_answers = sorted(prepositions)
    import random
    rng = random.Random(args.seed)
    sample_options = {}
    if args.n_options and args.n_options < len(all_answers):
        for idx in range(n_samples):
            gt = dataset[idx]["preposition"]
            distractors = [a for a in all_answers if a != gt]
            chosen = rng.sample(distractors, min(args.n_options - 1, len(distractors)))
            sample_options[idx] = sorted([gt] + chosen)

    for idx in tqdm(range(n_samples), desc="Text recovery"):
        sample = dataset[idx]
        objects = sample["objects"]
        gt = sample["preposition"]
        gt_token_id = answer_token_ids[gt]

        # Per-sample options (for n-options mode) or all prepositions
        eval_options = sample_options.get(idx, all_answers)

        # Build question
        if args.question_override:
            base_q = args.question_override
        elif "question_text" in sample and sample["question_text"]:
            base_q = sample["question_text"]
        else:
            base_q = create_question(
                objects, prepositions,
                question_format="free",
                task_type=args.task_type,
            )

        if args.question_format == "open":
            question = base_q + " Answer with a single word."
        elif args.question_format == "forced_choice":
            question = base_q + f" Answer only with {', '.join(eval_options)}."
        else:
            question = base_q

        # 1. Clean run
        inputs_clean = prepare_inputs(processor, sample["image"], question).to(model.device)
        _, clean_logits = collect_hidden_states(model, inputs_clean, [])

        # Check clean correctness (constrained over eval_options)
        option_logits_clean = {}
        for prep in eval_options:
            option_logits_clean[prep] = clean_logits[answer_token_ids[prep]].item()
        clean_choice = max(option_logits_clean, key=option_logits_clean.get)

        if clean_choice != gt:
            counts["skipped"] += 1
            continue

        clean_probs = torch.softmax(clean_logits, dim=-1)
        p_clean = clean_probs[gt_token_id].item()

        # Token ranges from clean input
        ranges_clean = find_token_ranges(inputs_clean.input_ids, processor.tokenizer)
        txt_start_c, txt_end_c = ranges_clean["text"]
        text_indices_clean = list(range(txt_start_c, txt_end_c))

        # Collect clean hidden states at all layers
        states_clean, _ = collect_hidden_states(model, inputs_clean, all_layers)

        # 2. Corrupted run (black image)
        w, h = sample["image"].size
        if args.corruption == "black":
            corrupt_image = Image.new("RGB", (w, h), (0, 0, 0))
        elif args.corruption == "white":
            corrupt_image = Image.new("RGB", (w, h), (255, 255, 255))
        else:  # noise
            arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            corrupt_image = Image.fromarray(arr)

        inputs_corrupt = prepare_inputs(processor, corrupt_image, question).to(model.device)
        _, corrupt_logits = collect_hidden_states(model, inputs_corrupt, [])
        corrupt_probs = torch.softmax(corrupt_logits, dim=-1)
        p_corrupt = corrupt_probs[gt_token_id].item()

        # Constrained choice on corrupted (over eval_options)
        option_logits_corrupt = {}
        for prep in eval_options:
            option_logits_corrupt[prep] = corrupt_logits[answer_token_ids[prep]].item()
        corrupt_choice = max(option_logits_corrupt, key=option_logits_corrupt.get)

        # Token ranges from corrupted input (may differ due to image size)
        ranges_corrupt = find_token_ranges(inputs_corrupt.input_ids, processor.tokenizer)
        txt_start_x, txt_end_x = ranges_corrupt["text"]
        text_indices_corrupt = list(range(txt_start_x, txt_end_x))

        # 3. Patched run: black image + ALL text states from clean at ALL layers
        patched_logits = run_with_text_patch(
            model, inputs_corrupt, states_clean,
            text_indices_corrupt, all_layers,
            source_text_indices=text_indices_clean,
        )
        patched_probs = torch.softmax(patched_logits, dim=-1)
        p_patched = patched_probs[gt_token_id].item()

        # Constrained choice on patched (over eval_options)
        option_logits_patched = {}
        for prep in eval_options:
            option_logits_patched[prep] = patched_logits[answer_token_ids[prep]].item()
        patched_choice = max(option_logits_patched, key=option_logits_patched.get)

        recovered = patched_choice == gt
        counts["recovered" if recovered else "not_recovered"] += 1

        # Top-5 argmax over full vocabulary for each condition
        def top5(logits):
            vals, ids = torch.topk(logits, 5)
            return [{"token": processor.tokenizer.decode([tid]).strip(),
                     "token_id": tid.item(),
                     "logit": round(v.item(), 4)}
                    for v, tid in zip(vals, ids)]

        results_per_sample.append({
            "idx": idx,
            "gt": gt,
            "clean_choice": clean_choice,
            "corrupt_choice": corrupt_choice,
            "patched_choice": patched_choice,
            "p_clean": round(p_clean, 6),
            "p_corrupt": round(p_corrupt, 6),
            "p_patched": round(p_patched, 6),
            "recovered": recovered,
            "option_logits_clean": {k: round(v, 4) for k, v in option_logits_clean.items()},
            "option_logits_corrupt": {k: round(v, 4) for k, v in option_logits_corrupt.items()},
            "option_logits_patched": {k: round(v, 4) for k, v in option_logits_patched.items()},
            "top5_clean": top5(clean_logits),
            "top5_corrupt": top5(corrupt_logits),
            "top5_patched": top5(patched_logits),
        })

        del states_clean
        torch.cuda.empty_cache()

    # Summary
    n_eval = counts["recovered"] + counts["not_recovered"]
    print(f"\n{'='*60}")
    print(f"Text Recovery Results ({n_eval} samples evaluated, {args.corruption} corruption)")
    print(f"{'='*60}")
    print(f"  Recovered:     {counts['recovered']:>4}  ({counts['recovered']/max(n_eval,1)*100:.1f}%)")
    print(f"  Not recovered: {counts['not_recovered']:>4}  ({counts['not_recovered']/max(n_eval,1)*100:.1f}%)")
    print(f"  Skipped:       {counts['skipped']:>4}  (model wrong on clean)")

    if results_per_sample:
        mean_p_clean = np.mean([r["p_clean"] for r in results_per_sample])
        mean_p_corrupt = np.mean([r["p_corrupt"] for r in results_per_sample])
        mean_p_patched = np.mean([r["p_patched"] for r in results_per_sample])
        print(f"\n  Mean P(correct):")
        print(f"    Clean (real image):     {mean_p_clean:.4f}")
        print(f"    Corrupted (no image):   {mean_p_corrupt:.4f}")
        print(f"    Patched (text restore): {mean_p_patched:.4f}")
        print(f"    Recovery: {(mean_p_patched - mean_p_corrupt) / max(mean_p_clean - mean_p_corrupt, 1e-6):.1%}")

    # Save
    if not args.no_save:
        fmt_suffix = f"_{args.question_format}" if args.question_format != "forced_choice" else ""
        task_suffix = f"_{args.task_type}" if args.task_type != "spatial_relative" else ""
        output_dir = (
            Path(__file__).parent.parent / "results" / args.results_subdir
            / f"{args.dataset_name}{fmt_suffix}{task_suffix}_{args.corruption}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "counts": counts,
            "n_evaluated": n_eval,
            "n_samples": n_samples,
            "dataset_name": args.dataset_name,
            "task_type": args.task_type,
            "question_format": args.question_format,
            "corruption": args.corruption,
            "per_sample": results_per_sample,
        }
        json_path = output_dir / "text_recovery_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
