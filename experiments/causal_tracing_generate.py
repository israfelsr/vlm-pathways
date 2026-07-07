"""
    python experiments/causal_tracing_generate.py --config configs/default.yaml \
        --dataset-name coco_recognition_pairs --prefill "It is a" \
        --use-question-text --n-pairs 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vlm_spatial.config import load_config
from vlm_spatial.data import (
    create_question,
    find_token_ranges,
    load_dataset,
)
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


def generate_with_patch(model, processor, inputs, clean_states, patch_layer,
                        patch_token_indices, source_token_indices=None,
                        max_new_tokens=10):
    """Generate with patching at a SINGLE layer during prompt processing.

    Args:
        patch_token_indices: indices in the TARGET (corrupted) run to write to
        source_token_indices: indices in the SOURCE (clean) states to read from
                              (default: same as patch_token_indices)
    """
    if source_token_indices is None:
        source_token_indices = patch_token_indices
    hooks = []
    prompt_len = inputs.input_ids.shape[1]

    def patch_hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)
        if hs.shape[1] != prompt_len:
            return output
        patched = hs.clone()
        clean = clean_states[patch_layer]
        for src_idx, tgt_idx in zip(source_token_indices, patch_token_indices):
            patched[0, tgt_idx, :] = clean[0, src_idx, :]
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    lm_layers = get_language_layers(model)
    handle = lm_layers[patch_layer].register_forward_hook(patch_hook)
    hooks.append(handle)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
    finally:
        for h in hooks:
            h.remove()

    generated_ids = output[0, prompt_len:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def extract_first_word(text):
    text = text.strip().lower()
    for prefix in ["it is a ", "it is an ", "it's a ", "a ", "an ", "the "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    words = text.split()
    if not words:
        return ""
    if len(words) >= 2 and words[1] in ("phone", "bat", "dog", "drier", "dryer", "control"):
        return f"{words[0]} {words[1]}"
    return words[0]


def outputs_match(text_a, text_b):
    w_a = extract_first_word(text_a)
    w_b = extract_first_word(text_b)
    if not w_a or not w_b:
        return False
    if w_a == w_b:
        return True
    if w_a in w_b or w_b in w_a:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Paired causal tracing with generate-based evaluation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--layer-step", type=int, default=1)
    parser.add_argument("--task-type", type=str, default="spatial_relative")
    parser.add_argument("--question-format", type=str, default="forced_choice")
    parser.add_argument("--prefill", type=str, default=None)
    parser.add_argument("--use-question-text", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--results-subdir", type=str, default="causal_tracing_gen")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model, processor = load_model(config["model_path"])
    dataset, prepositions = load_dataset(
        config["dataset_path"], args.dataset_name, task_type=args.task_type
    )

    n_layers = len(get_language_layers(model))
    layers_to_trace = list(range(0, n_layers, args.layer_step))

    # Build pair index
    pair_index = {}
    for i in range(len(dataset)):
        pid = dataset[i]["pair_id"]
        if pid not in pair_index:
            pair_index[pid] = []
        pair_index[pid].append(i)

    n_pairs = min(args.n_pairs, max(pair_index.keys()) + 1)

    print(f"Tracing {len(layers_to_trace)} layers (step={args.layer_step})")
    print(f"Processing {n_pairs} pairs")
    print(f"Prefill: {args.prefill!r}")
    print(f"Groups: image, all_text, last_token")

    # Define groups after we see the first sample's token ranges
    GROUPS = ["image", "all_text", "last_token"]

    # Per-layer per-group: count of matches_clean
    layer_matches_a = {g: {l: 0 for l in layers_to_trace} for g in GROUPS}
    layer_matches_b = {g: {l: 0 for l in layers_to_trace} for g in GROUPS}
    layer_total = {g: {l: 0 for l in layers_to_trace} for g in GROUPS}

    n_evaluated = 0
    n_skipped = 0
    sample_results = []

    for pair_id in tqdm(range(n_pairs), desc="Causal tracing (generate)"):
        indices = pair_index.get(pair_id, [])
        if len(indices) != 2:
            n_skipped += 1
            continue

        sample_a = dataset[indices[0]]
        sample_b = dataset[indices[1]]
        gt_a = sample_a["preposition"]
        gt_b = sample_b["preposition"]

        if args.use_question_text and "question_text" in sample_a:
            question = sample_a["question_text"]
        else:
            question = create_question(
                sample_a["objects"], prepositions,
                question_format=args.question_format,
                task_type=args.task_type,
            )

        inputs_a = prepare_inputs(processor, sample_a["image"], question,
                                  prefill=args.prefill).to(model.device)
        inputs_b = prepare_inputs(processor, sample_b["image"], question,
                                  prefill=args.prefill).to(model.device)

        # Token ranges from both A and B
        ranges_a = find_token_ranges(inputs_a.input_ids, processor.tokenizer)
        ranges_b = find_token_ranges(inputs_b.input_ids, processor.tokenizer)

        seq_len_a = inputs_a.input_ids.shape[1]
        seq_len_b = inputs_b.input_ids.shape[1]

        # Skip if image token counts differ (can't patch image group)
        # But text and last_token can still be patched using B's ranges
        # since the question is the same — text tokens are at different absolute
        # positions but same relative positions after the image block.

        # Build group indices: source (A) and target (B) separately
        group_indices_a = {}  # indices into A's hidden states
        group_indices_b = {}  # indices into B's hidden states (for patching)

        # Text tokens: same question, same tokens, just shifted by image count
        txt_s_a, txt_e_a = ranges_a["text"]
        txt_s_b, txt_e_b = ranges_b["text"]
        n_text_a = txt_e_a - txt_s_a
        n_text_b = txt_e_b - txt_s_b
        n_text = min(n_text_a, n_text_b)  # should be equal for same question
        group_indices_a["all_text"] = list(range(txt_s_a, txt_s_a + n_text))
        group_indices_b["all_text"] = list(range(txt_s_b, txt_s_b + n_text))

        # Last token
        group_indices_a["last_token"] = [seq_len_a - 1]
        group_indices_b["last_token"] = [seq_len_b - 1]

        # Image: only if same number of image tokens
        if ranges_a["image"] is not None and ranges_b["image"] is not None:
            img_s_a, img_e_a = ranges_a["image"]
            img_s_b, img_e_b = ranges_b["image"]
            if (img_e_a - img_s_a) == (img_e_b - img_s_b):
                group_indices_a["image"] = list(range(img_s_a, img_e_a))
                group_indices_b["image"] = list(range(img_s_b, img_e_b))

        # 1. Generate clean outputs
        prompt_len_a = inputs_a.input_ids.shape[1]
        with torch.no_grad():
            out_a = model.generate(**inputs_a, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_a = processor.tokenizer.decode(out_a[0, prompt_len_a:], skip_special_tokens=True).strip()

        prompt_len_b = inputs_b.input_ids.shape[1]
        with torch.no_grad():
            out_b = model.generate(**inputs_b, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_b = processor.tokenizer.decode(out_b[0, prompt_len_b:], skip_special_tokens=True).strip()

        word_a = extract_first_word(gen_a)
        word_b = extract_first_word(gen_b)

        # Skip if both generate the same thing
        if word_a == word_b:
            n_skipped += 1
            continue

        # 2. Collect A's hidden states at all layers
        layers_to_collect = sorted(set(layers_to_trace))
        states_a, _ = collect_hidden_states(model, inputs_a, layers_to_collect)

        # 3. For each group and layer, patch and generate
        pair_layer_results = {}
        for group_name in GROUPS:
            if group_name not in group_indices_b:
                continue
            tgt_indices = group_indices_b[group_name]
            src_indices = group_indices_a[group_name]

            for layer_idx in layers_to_trace:
                patched_gen = generate_with_patch(
                    model, processor, inputs_b, states_a,
                    patch_layer=layer_idx,
                    patch_token_indices=tgt_indices,
                    source_token_indices=src_indices,
                    max_new_tokens=args.max_new_tokens,
                )
                word_patched = extract_first_word(patched_gen)

                matches_a = bool(word_patched and (word_patched == word_a or word_a in word_patched or word_patched in word_a))
                matches_b = bool(word_patched and (word_patched == word_b or word_b in word_patched or word_patched in word_b))

                layer_matches_a[group_name][layer_idx] += int(matches_a)
                layer_matches_b[group_name][layer_idx] += int(matches_b)
                layer_total[group_name][layer_idx] += 1

                pair_layer_results[f"{group_name}_L{layer_idx}"] = {
                    "generated": patched_gen,
                    "word": word_patched,
                    "matches_a": matches_a,
                    "matches_b": matches_b,
                }

        n_evaluated += 1

        sample_results.append({
            "pair_id": pair_id,
            "gt_a": gt_a,
            "gt_b": gt_b,
            "gen_a": gen_a,
            "gen_b": gen_b,
            "word_a": word_a,
            "word_b": word_b,
            "question": question,
        })

        del states_a
        torch.cuda.empty_cache()

        if (pair_id + 1) % 10 == 0:
            print(f"  Pair {pair_id + 1}/{n_pairs} | evaluated: {n_evaluated} | "
                  f"A: {gen_a[:20]} | B: {gen_b[:20]}")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"Paired Causal Tracing (Generate) Results")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Evaluated: {n_evaluated}, Skipped: {n_skipped}")
    print(f"{'='*60}")

    # Build heatmaps: fraction matching A's answer (= restoration)
    heatmap = {}
    for group_name in GROUPS:
        heatmap[group_name] = {}
        print(f"\n  {group_name}:")
        print(f"  {'Layer':>6} {'Match A':>9} {'Match B':>9} {'Neither':>9} {'Total':>7} {'A%':>7}")
        print(f"  {'-'*50}")
        for l in layers_to_trace:
            total = layer_total[group_name][l]
            ma = layer_matches_a[group_name][l]
            mb = layer_matches_b[group_name][l]
            neither = total - ma - mb
            rate = ma / total if total > 0 else 0
            heatmap[group_name][l] = round(rate, 4)
            if l % 5 == 0:
                print(f"  {l:>6} {ma:>9} {mb:>9} {neither:>9} {total:>7} {rate:>7.1%}")

    results = {
        "heatmap": heatmap,
        "group_names": GROUPS,
        "layers": layers_to_trace,
        "n_pairs": n_pairs,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
        "dataset_name": args.dataset_name,
        "metric": "generate_matches_clean",
        "sample_results": sample_results,
    }

    if not args.no_save:
        output_dir = (
            Path(__file__).parent.parent / "results" / args.results_subdir
            / args.dataset_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "causal_tracing_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
