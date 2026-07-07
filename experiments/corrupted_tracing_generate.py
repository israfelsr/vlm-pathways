"""
    python experiments/corrupted_tracing_generate.py --config configs/default.yaml \
        --dataset-name coco_recognition --corruption black --n-samples 100
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vlm_spatial.config import load_config
from vlm_spatial.data import find_token_ranges, load_dataset
from vlm_spatial.hooks import get_language_layers
from vlm_spatial.model import load_model


def prepare_inputs(processor, image, question, prefill=None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    if prefill:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": prefill}],
        })
        text = processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True,
        )
    else:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )


def generate_with_patch(model, processor, inputs, clean_states, patch_layer,
                        patch_token_indices, max_new_tokens=20):
    """Generate text with patching active during the prompt processing step.

    The patch hook only fires during the first forward pass (full prompt).
    During autoregressive generation steps (single token), the hook checks
    sequence length and skips if it's a generation step.
    """
    hooks = []
    prompt_len = inputs.input_ids.shape[1]

    def patch_hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)

        # Only patch during prompt processing (full sequence), not generation steps
        if hs.shape[1] != prompt_len:
            return output

        patched = hs.clone()
        clean = clean_states[patch_layer]
        for idx in patch_token_indices:
            patched[0, idx, :] = clean[0, idx, :]

        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    lm_layers = get_language_layers(model)
    handle = lm_layers[patch_layer].register_forward_hook(patch_hook)
    hooks.append(handle)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        for h in hooks:
            h.remove()

    generated_ids = output.sequences[0, prompt_len:]
    generated_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # Also get first token probability for comparison
    if output.scores:
        first_logits = output.scores[0][0]
        first_probs = torch.softmax(first_logits, dim=-1)
    else:
        first_probs = None

    return generated_text, first_probs


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

    return hidden_states


def contains_answer(generated_text, ground_truth):
    """Check if generated text contains the ground truth answer."""
    return bool(re.search(rf'\b{re.escape(ground_truth)}\b', generated_text, re.IGNORECASE))


def get_task_type(ds_name):
    if "localization" in ds_name or ds_name.endswith("_one"):
        return "spatial_absolute"
    elif "recognition" in ds_name:
        return "recognition"
    return "spatial_relative"


def generate_corrupted_image(corruption_type, w, h, image_path=None):
    from PIL import Image as PILImage
    if corruption_type == "black":
        return PILImage.new("RGB", (w, h), (0, 0, 0))
    elif corruption_type == "white":
        return PILImage.new("RGB", (w, h), (255, 255, 255))
    elif corruption_type == "file":
        img = PILImage.open(image_path).convert("RGB")
        return img.resize((w, h), PILImage.BILINEAR)
    else:
        raise ValueError(f"Unsupported corruption: {corruption_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Corrupted tracing with generate-based evaluation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--corruption", type=str, default="black",
                        choices=["black", "white", "file"])
    parser.add_argument("--corruption-image", type=str, default=None)
    parser.add_argument("--layer-step", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--results-subdir", type=str, default="corrupted_tracing_gen")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model, processor = load_model(config["model_path"])
    dataset, prepositions = load_dataset(
        config["dataset_path"], args.dataset_name,
    )

    n_layers = len(get_language_layers(model))
    layers_to_trace = list(range(0, n_layers, args.layer_step))
    n_samples = min(args.n_samples, len(dataset))

    # Check if dataset has custom questions
    has_custom_questions = "question_text" in dataset.column_names
    prefill = "It is a" if has_custom_questions else None

    print(f"Tracing {n_layers} layers (step={args.layer_step}): {layers_to_trace[0]}-{layers_to_trace[-1]}")
    print(f"Processing {n_samples} samples (corruption={args.corruption})")
    print(f"Custom questions: {has_custom_questions}")
    print(f"Prefill: {prefill!r}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Metric: patched output matches clean output (first word)")

    def extract_first_word(text):
        """Extract first meaningful word from generated text."""
        text = text.strip().lower()
        # Remove common prefixes
        for prefix in ["it is a ", "it is an ", "it's a ", "it's an ", "a ", "an ", "the "]:
            if text.startswith(prefix):
                text = text[len(prefix):]
        # Take first word (or two for compound nouns)
        words = text.split()
        if not words:
            return ""
        # Keep two words if second word is a common noun continuation
        if len(words) >= 2 and words[1] in ("phone", "bat", "dog", "drier", "dryer", "control"):
            return f"{words[0]} {words[1]}"
        return words[0]

    def outputs_match(text_a, text_b):
        """Check if two generated outputs refer to the same thing."""
        w_a = extract_first_word(text_a)
        w_b = extract_first_word(text_b)
        if not w_a or not w_b:
            return False
        # Exact match
        if w_a == w_b:
            return True
        # One contains the other (e.g., "kite" in "kites")
        if w_a in w_b or w_b in w_a:
            return True
        return False

    # Per-layer results
    layer_matches_clean = {l: 0 for l in layers_to_trace}
    layer_matches_corrupt = {l: 0 for l in layers_to_trace}
    layer_total = {l: 0 for l in layers_to_trace}
    n_evaluated = 0
    n_skipped = 0
    sample_results = []

    for idx in tqdm(range(n_samples), desc=f"Generate tracing ({args.corruption})"):
        sample = dataset[idx]
        image = sample["image"]
        gt = sample["preposition"]

        # Use custom question if available
        if has_custom_questions:
            question = sample["question_text"]
        else:
            from vlm_spatial.data import create_question
            question = create_question(
                sample["objects"], prepositions,
                question_format="free", task_type=get_task_type(args.dataset_name),
            )

        # Prepare inputs (with prefill if applicable)
        inputs_clean = prepare_inputs(processor, image, question, prefill=prefill).to(model.device)

        corrupted_image = generate_corrupted_image(
            args.corruption, image.width, image.height,
            image_path=args.corruption_image,
        )
        inputs_corrupt = prepare_inputs(processor, corrupted_image, question, prefill=prefill).to(model.device)

        # Token ranges (from clean input)
        ranges = find_token_ranges(inputs_clean.input_ids, processor.tokenizer)
        txt_start, txt_end = ranges["text"]
        text_indices = list(range(txt_start, txt_end))

        # 1. Clean generate
        with torch.no_grad():
            prompt_len = inputs_clean.input_ids.shape[1]
            clean_output = model.generate(
                **inputs_clean, max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            clean_text = processor.tokenizer.decode(
                clean_output[0, prompt_len:], skip_special_tokens=True
            ).strip()

        # 2. Corrupted generate
        with torch.no_grad():
            prompt_len = inputs_corrupt.input_ids.shape[1]
            corrupt_output = model.generate(
                **inputs_corrupt, max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            corrupt_text = processor.tokenizer.decode(
                corrupt_output[0, prompt_len:], skip_special_tokens=True
            ).strip()

        # Skip if clean and corrupt give the same answer (no signal to restore)
        if outputs_match(clean_text, corrupt_text):
            n_skipped += 1
            continue

        # 3. Collect clean hidden states
        clean_states = collect_hidden_states(model, inputs_clean, layers_to_trace)

        # 4. For each layer, generate with text patching
        sample_layer_results = {}
        for layer_idx in layers_to_trace:
            patched_text, _ = generate_with_patch(
                model, processor, inputs_corrupt, clean_states,
                patch_layer=layer_idx,
                patch_token_indices=text_indices,
                max_new_tokens=args.max_new_tokens,
            )

            matches_clean = outputs_match(patched_text, clean_text)
            matches_corrupt = outputs_match(patched_text, corrupt_text)
            layer_matches_clean[layer_idx] += int(matches_clean)
            layer_matches_corrupt[layer_idx] += int(matches_corrupt)
            layer_total[layer_idx] += 1
            sample_layer_results[layer_idx] = {
                "generated": patched_text,
                "matches_clean": matches_clean,
                "matches_corrupt": matches_corrupt,
            }

        n_evaluated += 1

        sample_results.append({
            "idx": idx,
            "ground_truth": gt,
            "question": question,
            "clean_text": clean_text,
            "clean_first_word": extract_first_word(clean_text),
            "corrupt_text": corrupt_text,
            "corrupt_first_word": extract_first_word(corrupt_text),
            "layer_results": {str(l): r for l, r in sample_layer_results.items()},
        })

        del clean_states
        torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            print(f"  Sample {idx + 1}/{n_samples} | evaluated: {n_evaluated} | "
                  f"clean: {clean_text[:30]} | corrupt: {corrupt_text[:30]}")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"Generate-based Corrupted Tracing Results")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Corruption: {args.corruption}")
    print(f"  Evaluated: {n_evaluated}")
    print(f"  Skipped: {n_skipped} (clean == corrupt)")
    print(f"  Metric: patched output matches clean output")
    print(f"{'='*60}")

    print(f"\n{'Layer':>6} {'Clean':>8} {'Corrupt':>9} {'Neither':>9} {'Total':>7} {'Clean%':>8}")
    print("-" * 50)
    for l in layers_to_trace:
        total = layer_total[l]
        mc = layer_matches_clean[l]
        mr = layer_matches_corrupt[l]
        neither = total - mc - mr
        rate = mc / total if total > 0 else 0
        print(f"{l:>6} {mc:>8} {mr:>9} {neither:>9} {total:>7} {rate:>8.1%}")

    # Build results dict
    heatmap = {}
    heatmap["all_text"] = {}
    for l in layers_to_trace:
        total = layer_total[l]
        heatmap["all_text"][l] = round(layer_matches_clean[l] / total, 4) if total > 0 else 0.0

    heatmap_corrupt = {}
    heatmap_corrupt["all_text"] = {}
    for l in layers_to_trace:
        total = layer_total[l]
        heatmap_corrupt["all_text"][l] = round(layer_matches_corrupt[l] / total, 4) if total > 0 else 0.0

    results = {
        "heatmap": heatmap,
        "heatmap_corrupt": heatmap_corrupt,
        "group_names": ["all_text"],
        "layers": layers_to_trace,
        "n_samples": n_samples,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
        "corruption_type": args.corruption,
        "dataset_name": args.dataset_name,
        "metric": "generate_matches_clean",
        "sample_results": sample_results,
    }

    if not args.no_save:
        output_dir = (
            Path(__file__).parent.parent / "results" / args.results_subdir
            / f"{args.dataset_name}_{args.corruption}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
