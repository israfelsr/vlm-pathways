"""
    python experiments/text_dominance.py --config configs/default.yaml \
        --dataset-name controlled_shapes_pairs --n-pairs 50
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


def run_with_text_patch(model, inputs, clean_states, text_indices, layers,
                        source_text_indices=None):
    """Run forward pass replacing text hidden states from clean run at ALL layers.

    Args:
        source_text_indices: Indices to read FROM in clean_states. If None, same as text_indices.
            Set differently when A and B have different sequence lengths.
    """
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


def generate_with_text_patch(model, processor, inputs, clean_states, text_indices,
                             all_layers, max_new_tokens=10, source_text_indices=None):
    """Generate with text patching active during prompt processing.

    Args:
        source_text_indices: Indices to read FROM in clean_states. If None, same as text_indices.
    """
    if source_text_indices is None:
        source_text_indices = text_indices
    hooks = []
    prompt_len = inputs.input_ids.shape[1]

    for layer_idx in all_layers:
        def make_hook(l_idx):
            def hook_fn(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                if hs.dim() == 2:
                    hs = hs.unsqueeze(0)
                # Only patch during prompt processing, not generation steps
                if hs.shape[1] != prompt_len:
                    return output
                patched = hs.clone()
                clean = clean_states[l_idx]
                for src_idx, tgt_idx in zip(source_text_indices, text_indices):
                    patched[0, tgt_idx, :] = clean[0, src_idx, :]
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return hook_fn

        lm_layers = get_language_layers(model)
        handle = lm_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
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
    generated_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()
    return generated_text


def extract_first_word(text):
    """Extract first meaningful word from generated text."""
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


# Synonym map for recognition correctness checks
RECOGNITION_SYNONYMS = {
    "bird": ["bird", "pigeon", "seagull", "turkey", "parrot", "sparrow",
             "crow", "eagle", "duck", "goose", "owl", "penguin",
             "hen", "rooster", "chicken"],
    "hair drier": ["hair drier", "hair dryer", "hairdryer", "blow dryer"],
    "cell phone": ["cell phone", "cellphone", "phone", "smartphone", "mobile"],
    "remote": ["remote", "remote control"],
    "hot dog": ["hot dog", "hotdog"],
    "baseball bat": ["baseball bat", "bat"],
    "cup": ["cup", "mug"],
    "dog": ["dog", "puppy"],
    "cat": ["cat", "kitten"],
    "laptop": ["laptop", "computer", "notebook"],
}


def matches_gt(generated_word, gt):
    """Check if generated word matches ground truth (with synonym matching)."""
    gt_lower = gt.lower()
    gen_lower = generated_word.lower()
    # Exact match
    if gen_lower == gt_lower:
        return True
    # Synonym match
    variants = RECOGNITION_SYNONYMS.get(gt_lower, [gt_lower])
    return gen_lower in [v.lower() for v in variants]


def main():
    parser = argparse.ArgumentParser(description="Text vs Image dominance test")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--task-type", type=str, default="spatial_relative",
                        choices=["spatial_relative", "spatial_absolute", "recognition"])
    parser.add_argument("--question-format", type=str, default="forced_choice",
                        choices=["forced_choice", "open", "free"])
    parser.add_argument("--generate", action="store_true",
                        help="Use generate mode instead of single-token logits. "
                             "Required for multi-word answers (recognition tasks).")
    parser.add_argument("--prefill", type=str, default=None,
                        help="Prefill assistant response (e.g., 'It is a')")
    parser.add_argument("--use-question-text", action="store_true",
                        help="Use question_text field from dataset instead of create_question()")
    parser.add_argument("--results-subdir", type=str, default="text_dominance")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model, processor = load_model(config["model_path"])
    dataset, prepositions = load_dataset(
        config["dataset_path"], args.dataset_name, task_type=args.task_type
    )

    n_layers = len(get_language_layers(model))
    all_layers = list(range(n_layers))

    # Build pair index
    pair_index = {}
    for i in range(len(dataset)):
        pid = dataset[i]["pair_id"]
        if pid not in pair_index:
            pair_index[pid] = []
        pair_index[pid].append(i)

    n_pairs = min(args.n_pairs, max(pair_index.keys()) + 1)

    results_per_pair = []
    counts = {"text_wins": 0, "image_wins": 0, "other": 0, "skipped": 0}

    use_generate = args.generate
    prefill = args.prefill
    use_question_text = args.use_question_text

    for pair_id in tqdm(range(n_pairs), desc="Text dominance"):
        indices = pair_index[pair_id]
        if len(indices) != 2:
            counts["skipped"] += 1
            continue

        sample_a = dataset[indices[0]]
        sample_b = dataset[indices[1]]
        objects = sample_a["objects"]
        gt_a = sample_a["preposition"]
        gt_b = sample_b["preposition"]

        if use_question_text and "question_text" in sample_a:
            question = sample_a["question_text"]
        else:
            question = create_question(
                objects, prepositions,
                question_format=args.question_format,
                task_type=args.task_type,
            )

        inputs_a = prepare_inputs(processor, sample_a["image"], question, prefill=prefill).to(model.device)
        inputs_b = prepare_inputs(processor, sample_b["image"], question, prefill=prefill).to(model.device)

        # Token ranges — compute for both A and B in case sequence lengths differ
        ranges_a = find_token_ranges(inputs_a.input_ids, processor.tokenizer)
        txt_start_a, txt_end_a = ranges_a["text"]
        text_indices_a = list(range(txt_start_a, txt_end_a))

        seq_len_a = inputs_a.input_ids.shape[1]
        seq_len_b = inputs_b.input_ids.shape[1]
        if seq_len_b != seq_len_a:
            ranges_b = find_token_ranges(inputs_b.input_ids, processor.tokenizer)
            txt_start_b, txt_end_b = ranges_b["text"]
            text_indices_b = list(range(txt_start_b, txt_end_b))
        else:
            text_indices_b = text_indices_a

        if use_generate:
            # --- Generate mode: compare generated text ---
            # 1. Clean generates
            prompt_len_a = inputs_a.input_ids.shape[1]
            with torch.no_grad():
                out_a = model.generate(**inputs_a, max_new_tokens=10, do_sample=False)
                gen_a = processor.tokenizer.decode(out_a[0, prompt_len_a:], skip_special_tokens=True).strip()

            prompt_len_b = inputs_b.input_ids.shape[1]
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=10, do_sample=False)
                gen_b = processor.tokenizer.decode(out_b[0, prompt_len_b:], skip_special_tokens=True).strip()

            word_a = extract_first_word(gen_a)
            word_b = extract_first_word(gen_b)

            # Skip if model gets either clean run wrong
            a_correct = matches_gt(word_a, gt_a)
            b_correct = matches_gt(word_b, gt_b)
            if not (a_correct and b_correct):
                counts["skipped"] += 1
                continue

            # 2. Collect A's hidden states
            states_a, _ = collect_hidden_states(model, inputs_a, all_layers)

            # 3. Generate with B's image + A's text
            patched_gen = generate_with_text_patch(
                model, processor, inputs_b, states_a, text_indices_b,
                all_layers, max_new_tokens=10,
                source_text_indices=text_indices_a,
            )
            word_patched = extract_first_word(patched_gen)

            if word_patched == word_a:
                outcome = "text_wins"
            elif word_patched == word_b:
                outcome = "image_wins"
            else:
                outcome = "other"
            counts[outcome] += 1

            pair_result = {
                "pair_id": pair_id,
                "gt_a": gt_a,
                "gt_b": gt_b,
                "gen_a": gen_a,
                "gen_b": gen_b,
                "word_a": word_a,
                "word_b": word_b,
                "patched_gen": patched_gen,
                "word_patched": word_patched,
                "outcome": outcome,
            }

        else:
            # --- Logit mode: compare first-token probabilities ---
            gt_a_id = processor.tokenizer.encode(gt_a, add_special_tokens=False)[0]
            gt_b_id = processor.tokenizer.encode(gt_b, add_special_tokens=False)[0]

            # 1. Clean runs
            _, logits_a = collect_hidden_states(model, inputs_a, [])
            clean_a_argmax = logits_a.argmax().item()

            _, logits_b = collect_hidden_states(model, inputs_b, [])
            clean_b_argmax = logits_b.argmax().item()

            # Skip if model gets either clean run wrong
            # Use constrained evaluation when prepositions are available
            if prepositions and len(prepositions) > 1:
                option_logits_a = {}
                option_logits_b = {}
                for prep in prepositions:
                    prep_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
                    option_logits_a[prep] = logits_a[prep_ids[0]].item()
                    option_logits_b[prep] = logits_b[prep_ids[0]].item()
                a_correct = max(option_logits_a, key=option_logits_a.get) == gt_a
                b_correct = max(option_logits_b, key=option_logits_b.get) == gt_b
            else:
                a_correct = (clean_a_argmax == gt_a_id or
                             processor.tokenizer.decode([clean_a_argmax]).strip().lower() ==
                             processor.tokenizer.decode([gt_a_id]).strip().lower())
                b_correct = (clean_b_argmax == gt_b_id or
                             processor.tokenizer.decode([clean_b_argmax]).strip().lower() ==
                             processor.tokenizer.decode([gt_b_id]).strip().lower())

            if not (a_correct and b_correct):
                counts["skipped"] += 1
                continue

            # 2. Collect A's hidden states
            states_a, _ = collect_hidden_states(model, inputs_a, all_layers)

            # 3. Run image B with text states from A
            patched_logits = run_with_text_patch(
                model, inputs_b, states_a, text_indices_b, all_layers,
                source_text_indices=text_indices_a,
            )
            patched_probs = torch.softmax(patched_logits, dim=-1)
            patched_argmax = patched_logits.argmax().item()
            patched_answer = processor.tokenizer.decode([patched_argmax]).strip().lower()

            p_a = patched_probs[gt_a_id].item()
            p_b = patched_probs[gt_b_id].item()

            clean_probs_b = torch.softmax(logits_b, dim=-1)
            p_b_clean_a = clean_probs_b[gt_a_id].item()
            p_b_clean_b = clean_probs_b[gt_b_id].item()

            # Determine outcome by comparing P(GT_A) vs P(GT_B)
            if p_a >= p_b:
                outcome = "text_wins"
            else:
                outcome = "image_wins"
            counts[outcome] += 1

            pair_result = {
                "pair_id": pair_id,
                "gt_a": gt_a,
                "gt_b": gt_b,
                "patched_answer": patched_answer,
                "p_clean_b_answer_a": round(p_b_clean_a, 6),
                "p_clean_b_answer_b": round(p_b_clean_b, 6),
                "p_patched_answer_a": round(p_a, 6),
                "p_patched_answer_b": round(p_b, 6),
                "outcome": outcome,
            }

        results_per_pair.append(pair_result)

        del states_a
        torch.cuda.empty_cache()

    # Summary
    n_eval = counts["text_wins"] + counts["image_wins"] + counts["other"]
    print(f"\n{'='*50}")
    print(f"Text Dominance Results ({n_eval} pairs evaluated)")
    print(f"{'='*50}")
    print(f"  Text wins:  {counts['text_wins']:>4}  ({counts['text_wins']/max(n_eval,1)*100:.1f}%)")
    print(f"  Image wins: {counts['image_wins']:>4}  ({counts['image_wins']/max(n_eval,1)*100:.1f}%)")
    print(f"  Other:      {counts['other']:>4}  ({counts['other']/max(n_eval,1)*100:.1f}%)")
    print(f"  Skipped:    {counts['skipped']:>4}")

    # Mean probabilities + shift from patching (logit mode only)
    if results_per_pair and "p_patched_answer_a" in results_per_pair[0]:
        mean_p_text = np.mean([r["p_patched_answer_a"] for r in results_per_pair])
        mean_p_image = np.mean([r["p_patched_answer_b"] for r in results_per_pair])
        mean_p_clean_a = np.mean([r["p_clean_b_answer_a"] for r in results_per_pair])
        mean_p_clean_b = np.mean([r["p_clean_b_answer_b"] for r in results_per_pair])
        print(f"\n  --- Before patching (image B, no intervention) ---")
        print(f"  Mean P(A's answer): {mean_p_clean_a:.3f}  (should be low)")
        print(f"  Mean P(B's answer): {mean_p_clean_b:.3f}  (should be high)")
        print(f"\n  --- After patching (image B + text from A) ---")
        print(f"  Mean P(A's answer): {mean_p_text:.3f}  (text effect)")
        print(f"  Mean P(B's answer): {mean_p_image:.3f}  (image effect)")
        print(f"\n  Shift in P(A): {mean_p_text - mean_p_clean_a:+.3f}")
        print(f"  Shift in P(B): {mean_p_image - mean_p_clean_b:+.3f}")

    # Save
    if not args.no_save:
        output_dir = (
            Path(__file__).parent.parent / "results" / args.results_subdir
            / args.dataset_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "counts": counts,
            "n_evaluated": n_eval,
            "n_pairs": n_pairs,
            "per_pair": results_per_pair,
            "dataset_name": args.dataset_name,
            "task_type": args.task_type,
            "question_format": args.question_format,
        }
        json_path = output_dir / "text_dominance_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
