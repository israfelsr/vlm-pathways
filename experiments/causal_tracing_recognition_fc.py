"""
    python experiments/causal_tracing_recognition_fc.py --config configs/default.yaml \
        --dataset-name coco_recognition_pairs --n-pairs 99999 \
        --results-subdir qwen/causal_tracing
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vlm_spatial import load_model, load_dataset, load_config, find_token_ranges, find_cluster_ranges
from vlm_spatial.hooks import get_language_layers
from experiments.causal_tracing import (
    prepare_inputs,
    collect_hidden_states,
    run_with_patch,
    define_token_groups,
    _get_logits,
    save_results,
    print_heatmap,
)


def main():
    parser = argparse.ArgumentParser(
        description="Causal tracing for recognition with 4-option forced choice"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="coco_recognition_pairs")
    parser.add_argument("--n-pairs", type=int, default=99999)
    parser.add_argument("--groups", type=str, default="image,all_text,last_token")
    parser.add_argument("--results-subdir", type=str, default="causal_tracing")
    parser.add_argument("--save-option-logits", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model, processor = load_model(config["model_path"])
    dataset, prepositions = load_dataset(
        config["dataset_path"], args.dataset_name, task_type="recognition"
    )

    all_answers = sorted(prepositions)
    n_layers = len(get_language_layers(model))
    layers_to_trace = list(range(n_layers))
    group_specs = [g.strip() for g in args.groups.split(",")]

    # Build first-token IDs
    answer_token_ids = {}
    for ans in all_answers:
        token_ids = processor.tokenizer.encode(ans, add_special_tokens=False)
        answer_token_ids[ans] = token_ids[0]

    # Build pair index
    pair_index = {}
    for i in range(len(dataset)):
        pid = dataset[i]["pair_id"]
        if pid not in pair_index:
            pair_index[pid] = []
        pair_index[pid].append(i)

    total_pairs = max(pair_index.keys()) + 1
    n_pairs = min(args.n_pairs, total_pairs)

    # Pre-compute 4-option sets per pair (seeded)
    rng = random.Random(args.seed)
    pair_options = {}
    for pid in range(n_pairs):
        if pid not in pair_index or len(pair_index[pid]) != 2:
            continue
        gt_a = dataset[pair_index[pid][0]]["preposition"]
        gt_b = dataset[pair_index[pid][1]]["preposition"]
        distractors = [a for a in all_answers if a not in (gt_a, gt_b)]
        chosen = rng.sample(distractors, min(2, len(distractors)))
        pair_options[pid] = sorted([gt_a, gt_b] + chosen)

    print(f"Processing {n_pairs} pairs, 4-option forced choice")
    print(f"Tracing {n_layers} layers, groups: {group_specs}")

    # Accumulators
    all_scores = {}
    all_argmax = {}
    all_option_scores = {}
    per_pair_scores = []
    pair_metadata = []
    pair_gt_ids = []
    pair_corrupted_ids = []
    group_names = None
    n_skipped = 0

    for pair_id in tqdm(range(n_pairs), desc="Causal tracing (FC)"):
        if pair_id not in pair_index or len(pair_index[pair_id]) != 2:
            continue
        if pair_id not in pair_options:
            continue

        idx_a, idx_b = pair_index[pair_id]
        sample_a = dataset[idx_a]
        sample_b = dataset[idx_b]
        gt_a = sample_a["preposition"]
        gt_b = sample_b["preposition"]
        options = pair_options[pair_id]

        gt_token_id = answer_token_ids[gt_a]

        # Build forced-choice question from question_text
        base_q = sample_a.get("question_text", "What is in the image?")
        question = f"{base_q} Answer only with {', '.join(options)}."

        # Option token IDs for this pair's 4 options
        option_tids = {opt: answer_token_ids[opt] for opt in options}

        # Prepare inputs
        inputs_a = prepare_inputs(processor, sample_a["image"], question).to(model.device)
        inputs_b = prepare_inputs(processor, sample_b["image"], question).to(model.device)

        # Token ranges
        ranges_a = find_token_ranges(inputs_a.input_ids, processor.tokenizer)
        cluster_ranges_a = find_cluster_ranges(
            inputs_a.input_ids, processor.tokenizer,
            sample_a["objects"], ground_truth=gt_a, prepositions=set(options),
        )

        token_groups = define_token_groups(
            ranges_a, cluster_ranges_a, group_specs=group_specs,
        )
        seq_len_a = inputs_a.input_ids.shape[1]
        token_groups = [
            (name, [idx if idx >= 0 else seq_len_a + idx for idx in indices])
            for name, indices in token_groups
        ]

        # B's token groups if different sequence length
        seq_len_b = inputs_b.input_ids.shape[1]
        if seq_len_b != seq_len_a:
            ranges_b = find_token_ranges(inputs_b.input_ids, processor.tokenizer)
            cluster_ranges_b = find_cluster_ranges(
                inputs_b.input_ids, processor.tokenizer,
                sample_b["objects"], ground_truth=gt_b, prepositions=set(options),
            )
            token_groups_b = define_token_groups(
                ranges_b, cluster_ranges_b, group_specs=group_specs,
            )
            token_groups_b = [
                (name, [idx if idx >= 0 else seq_len_b + idx for idx in indices])
                for name, indices in token_groups_b
            ]
        else:
            token_groups_b = token_groups

        if group_names is None:
            group_names = [g[0] for g in token_groups]
            for name in group_names:
                all_scores[name] = {l: [] for l in layers_to_trace}
                all_argmax[name] = {l: [] for l in layers_to_trace}
                if args.save_option_logits:
                    # Track all 22 answers globally (not just this pair's 4)
                    all_option_scores[name] = {
                        l: {opt: [] for opt in all_answers} for l in layers_to_trace
                    }

        # 1. Clean run
        layers_to_collect = sorted(set(layers_to_trace) | {0})
        clean_states, clean_logits = collect_hidden_states(model, inputs_a, layers_to_collect)

        # Constrained check over 4 options
        clean_option_logits = {opt: clean_logits[answer_token_ids[opt]].item() for opt in options}
        clean_choice = max(clean_option_logits, key=clean_option_logits.get)
        if clean_choice != gt_a:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  Skipping pair {pair_id}: clean={clean_choice}, gt={gt_a}")
            continue

        clean_probs = torch.softmax(clean_logits, dim=-1)
        p_clean = clean_probs[gt_token_id].item()

        # 2. Corrupted run
        corrupted_logits = _get_logits(model, inputs_b)
        corrupted_probs = torch.softmax(corrupted_logits, dim=-1)
        p_corrupted = corrupted_probs[gt_token_id].item()
        corrupted_argmax = corrupted_logits.argmax().item()

        # Check corrupted side
        corrupt_option_logits = {opt: corrupted_logits[answer_token_ids[opt]].item() for opt in options}
        corrupt_choice = max(corrupt_option_logits, key=corrupt_option_logits.get)
        if corrupt_choice != gt_b:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  Skipping pair {pair_id}: corrupt={corrupt_choice}, gt_b={gt_b}")
            continue

        gap = p_clean - p_corrupted

        pair_metadata.append({
            "pair_id": pair_id,
            "gt_a": gt_a,
            "gt_b": gt_b,
            "options": options,
            "p_clean": round(p_clean, 6),
            "p_corrupted": round(p_corrupted, 6),
            "gap": round(gap, 6),
        })

        pair_gt_ids.append(gt_token_id)
        pair_corrupted_ids.append(corrupted_argmax)

        # 3. Patched runs
        pair_scores_entry = {}
        for (group_name, src_indices), (_, tgt_indices) in zip(token_groups, token_groups_b):
            pair_scores_entry[group_name] = {}
            for layer_idx in layers_to_trace:
                result = run_with_patch(
                    model, inputs_b, clean_states,
                    patch_layer=layer_idx,
                    patch_token_indices=tgt_indices,
                    gt_token_id=gt_token_id,
                    source_token_indices=src_indices,
                    option_token_ids=option_tids if args.save_option_logits else None,
                )

                if args.save_option_logits:
                    p_patched, patched_argmax, patched_opt_logits = result
                else:
                    p_patched, patched_argmax = result

                score = (p_patched - p_corrupted) / gap if abs(gap) > 1e-6 else 0.0

                all_scores[group_name][layer_idx].append(score)
                all_argmax[group_name][layer_idx].append(patched_argmax)
                pair_scores_entry[group_name][layer_idx] = round(score, 6)

                if args.save_option_logits:
                    for opt in options:
                        opt_gap = clean_option_logits[opt] - corrupt_option_logits[opt]
                        if abs(opt_gap) > 1e-6:
                            opt_score = (patched_opt_logits[opt] - corrupt_option_logits[opt]) / opt_gap
                        else:
                            opt_score = 0.0
                        all_option_scores[group_name][layer_idx][opt].append(opt_score)

        per_pair_scores.append({"pair_id": pair_id, "scores": pair_scores_entry})

        del clean_states
        torch.cuda.empty_cache()

    # Aggregate
    n_evaluated = len(pair_gt_ids)
    print(f"\nEvaluated {n_evaluated} pairs, skipped {n_skipped}")

    heatmap = {}
    heatmap_std = {}
    argmax_fractions = {}
    for group_name in group_names:
        heatmap[group_name] = {}
        heatmap_std[group_name] = {}
        argmax_fractions[group_name] = {}
        for layer_idx in layers_to_trace:
            scores = all_scores[group_name][layer_idx]
            heatmap[group_name][layer_idx] = float(np.mean(scores)) if scores else 0.0
            heatmap_std[group_name][layer_idx] = float(np.std(scores)) if scores else 0.0
            argmax_ids = all_argmax[group_name][layer_idx]
            if argmax_ids:
                n = len(argmax_ids)
                n_clean = sum(1 for a, gt in zip(argmax_ids, pair_gt_ids) if a == gt)
                n_corrupted = sum(1 for a, gt, corr in zip(argmax_ids, pair_gt_ids, pair_corrupted_ids)
                                  if a == corr and a != gt)
                argmax_fractions[group_name][layer_idx] = {
                    "clean": round(n_clean / n, 4),
                    "corrupted": round(n_corrupted / n, 4),
                    "other": round((n - n_clean - n_corrupted) / n, 4),
                }

    option_heatmaps = {}
    if all_option_scores:
        for group_name in all_option_scores:
            option_heatmaps[group_name] = {}
            for layer_idx in layers_to_trace:
                option_heatmaps[group_name][layer_idx] = {}
                for opt in all_option_scores[group_name][layer_idx]:
                    scores = all_option_scores[group_name][layer_idx][opt]
                    option_heatmaps[group_name][layer_idx][opt] = round(
                        float(np.mean(scores)) if scores else 0.0, 6
                    )

    results = {
        "heatmap": heatmap,
        "heatmap_std": heatmap_std,
        "argmax_fractions": argmax_fractions,
        "pair_metadata": pair_metadata,
        "per_pair_scores": per_pair_scores,
        "group_names": group_names,
        "layers": layers_to_trace,
        "n_pairs": n_pairs,
        "n_skipped": n_skipped,
        "n_evaluated": n_evaluated,
        "n_layers": n_layers,
        "dataset_name": args.dataset_name,
        "question_format": "forced_choice",
        "n_options": 4,
        "seed": args.seed,
    }
    if option_heatmaps:
        results["option_heatmaps"] = option_heatmaps

    print_heatmap(results)

    if not args.no_save:
        output_dir = (
            Path(__file__).parent.parent / "results" / args.results_subdir
            / f"{args.dataset_name}"
        )
        save_results(results, output_dir=output_dir)


if __name__ == "__main__":
    main()
