"""
    # Direct knockout (last→image)
    python experiments/02_attention_knockout/knockout_direct_path.py --config configs/default.yaml

    # Mediated knockout (all text→image)
    python experiments/02_attention_knockout/knockout_direct_path.py --config configs/default.yaml --knockout-type mediated

    # Cluster knockout (specific cluster→image)
    python experiments/02_attention_knockout/knockout_direct_path.py --config configs/default.yaml --knockout-type cluster --cluster object1
    python experiments/02_attention_knockout/knockout_direct_path.py --config configs/default.yaml --knockout-type cluster --cluster relation

    # With layer range
    python experiments/02_attention_knockout/knockout_direct_path.py --config configs/default.yaml --layers 23-28

    # Skip baseline (use saved results from experiments/baseline.py)
    python experiments/02_attention_knockout/knockout_direct_path.py --config configs/default.yaml --baseline-results results/baseline/controlled_clevr/diagnostic_samples.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from vlm_spatial import (
    load_model,
    load_dataset,
    create_question,
    find_token_ranges,
    find_cluster_ranges,
    install_last_to_image_block,
    install_text_to_image_block,
    install_cluster_to_image_block,
    install_last_to_text_block,
    install_last_to_cluster_block,
    normalize_cluster_indices,
    load_config,
    parse_layer_range,
)


def load_baseline_results(path):
    """Load baseline results and map field names.

    Handles two formats:
    1. Standard baseline (list of samples from baseline.py)
    2. Recognition baseline (dict with 'samples' key from baseline_recognition_pairs.py)

    Maps field names to knockout script conventions.
    """
    with open(path) as f:
        raw = json.load(f)

    # Handle dict-wrapped format (baseline_recognition_pairs.py)
    if isinstance(raw, dict) and "samples" in raw:
        raw = raw["samples"]

    mapped = []
    for r in raw:
        mapped.append(
            {
                "prediction": r.get("generated_text", r.get("generated", r.get("prediction", ""))),
                "ground_truth": r["ground_truth"],
                "correct": r.get("correct", r.get("correct_first_token_id",
                            r.get("correct_constrained", False))),
                "first_token": r.get("first_token", r.get("first_generated_token",
                               r.get("constrained_choice", ""))),
                "first_token_id": r.get(
                    "first_token_id", r.get("first_generated_id", -1)
                ),
                "gt_token_ids": r.get("gt_token_ids", []),
                "max_logit": r.get("max_logit", 0),
                "top5_tokens": r.get("top5_tokens", []),
                "top5_ids": r.get("top5_ids", []),
                "top5_logits": r.get("top5_logits", []),
                "gt_in_top5": r.get(
                    "gt_in_top5", r.get("correct_top5_token_id", False)
                ),
                "sample_idx": r.get("sample_idx", r.get("idx", 0)),
                "correct_constrained": r.get("correct_constrained"),
                "constrained_choice": r.get("constrained_choice"),
                "option_logits": r.get("option_logits"),
            }
        )
    return mapped


def evaluate_sample(
    model,
    processor,
    image,
    question,
    ground_truth,
    use_knockout=False,
    knockout_type="direct",
    knockout_fraction=1.0,
    layer_range=None,
    objects=None,
    cluster=None,
    prepositions=None,
    complement=False,
    verbose=False,
):
    """
    Evaluate a single sample with optional attention knockout.

    Uses model.generate() to handle multi-token answers (e.g., "behind" = "beh" + "ind").
    Hooks are applied during each forward pass inside generate().

    Args:
        model: Qwen3VL model
        processor: Processor
        image: PIL Image
        question: Question string
        ground_truth: Ground truth answer (e.g., "left", "right", "behind")
        use_knockout: Whether to apply attention masking
        knockout_type: "direct" (last→image), "mediated" (text→image), or "cluster"
        knockout_fraction: Fraction of image tokens to block (0.0 to 1.0)
        layer_range: Optional (start, end) for layer-specific knockout [start, end)
        objects: List [obj1, obj2] for cluster knockout
        cluster: Cluster name for knockout ("object1", "object2", "relation", "format")

    Returns:
        dict with prediction, correctness, and ground truth
    """
    # Prepare inputs
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

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_len = inputs.input_ids.shape[1]

    # Generate with optional knockout
    with torch.no_grad():
        if use_knockout:
            # Get token ranges for masking (based on prompt)
            ranges = find_token_ranges(inputs.input_ids, processor.tokenizer)

            if verbose:
                ids = inputs.input_ids[0]
                img_s, img_e = ranges["image"]
                txt_s, txt_e = ranges["text"]
                print(f"\n{'='*70}")
                print(f"DIAGNOSTIC — first sample")
                print(f"{'='*70}")
                print(f"Question: {question}")
                print(f"Ground truth: {ground_truth}")
                print(f"Objects: {objects}")
                print(f"Total tokens: {len(ids)}")
                print(f"Image tokens: [{img_s}, {img_e})  ({img_e - img_s} tokens)")
                print(f"Text tokens:  [{txt_s}, {txt_e})  ({txt_e - txt_s} tokens)")
                print(f"Last token:   {ranges['last']}  (resolves to {len(ids) - 1})")
                print(f"\nFull prompt (chat template):")
                print(text)
                print(f"\nText token content:")
                for i in range(txt_s, txt_e):
                    tok = processor.tokenizer.decode([ids[i]])
                    print(f"  [{i:4d}] {tok!r}")
                print(f"{'='*70}\n")

            # Install hooks based on knockout type
            if knockout_type == "direct":
                hooks, stats = install_last_to_image_block(
                    model, ranges, layer_range, knockout_fraction
                )
            elif knockout_type == "mediated":
                hooks, stats = install_text_to_image_block(
                    model, ranges, layer_range, knockout_fraction
                )
            elif knockout_type == "cluster":
                if objects is None or cluster is None:
                    raise ValueError(
                        "cluster knockout requires objects and cluster args"
                    )
                clusters = find_cluster_ranges(
                    inputs.input_ids,
                    processor.tokenizer,
                    objects,
                    ground_truth=ground_truth,
                    prepositions=prepositions,
                )
                # Resolve cluster spec (supports '+' for merging)
                if "+" in cluster:
                    names = cluster.split("+")
                    merged = []
                    for name in names:
                        spec = clusters[name]
                        if spec is None:
                            continue
                        if isinstance(spec, tuple):
                            merged.append(spec)
                        else:
                            merged.extend(spec)
                    cluster_spec = merged if merged else None
                else:
                    cluster_spec = clusters[cluster]

                # Complement: block everything EXCEPT this cluster
                if complement:
                    txt_start, txt_end = ranges["text"]
                    all_text = set(range(txt_start, txt_end))
                    keep = set(normalize_cluster_indices(cluster_spec))
                    block = sorted(all_text - keep)
                    if not hasattr(evaluate_sample, '_complement_printed'):
                        evaluate_sample._complement_printed = True
                        print(f"\n--- Complement Mode ---")
                        print(f"Text range: {txt_start}-{txt_end} ({txt_end - txt_start} tokens)")
                        print(f"Keep (can see image): {len(keep)} tokens")
                        print(f"Block (blocked from image): {len(block)} tokens")
                        print(f"---\n")
                    cluster_spec = [(i, i + 1) for i in block]

                hooks, stats = install_cluster_to_image_block(
                    model, ranges, cluster_spec, layer_range
                )
            elif knockout_type == "last_to_text":
                hooks, stats = install_last_to_text_block(
                    model, ranges, layer_range
                )
            elif knockout_type == "last_to_cluster":
                if objects is None or cluster is None:
                    raise ValueError(
                        "last_to_cluster knockout requires objects and cluster args"
                    )
                clusters = find_cluster_ranges(
                    inputs.input_ids,
                    processor.tokenizer,
                    objects,
                    ground_truth=ground_truth,
                    prepositions=prepositions,
                )
                # Resolve cluster spec (supports '+' for merging)
                if "+" in cluster:
                    names = cluster.split("+")
                    merged = []
                    for name in names:
                        spec = clusters[name]
                        if spec is None:
                            continue
                        if isinstance(spec, tuple):
                            merged.append(spec)
                        else:
                            merged.extend(spec)
                    cluster_spec = merged if merged else None
                else:
                    cluster_spec = clusters[cluster]
                hooks, stats = install_last_to_cluster_block(
                    model, ranges, cluster_spec, layer_range
                )
            else:
                raise ValueError(f"Unknown knockout_type: {knockout_type}")

            try:
                output = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            finally:
                # Always remove hooks
                for h in hooks:
                    h.remove()
                if verbose:
                    print(f"Hook stats: {stats}")
        else:
            # Baseline: no manipulation
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

    # Get generated sequence (excluding prompt)
    generated_ids = output.sequences[0, prompt_len:]
    generated_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # Get first generated token ID
    first_token_scores = output.scores[0]  # [batch, vocab]
    first_token_id = generated_ids[0].item()
    first_token = processor.tokenizer.decode(
        [first_token_id], skip_special_tokens=True
    ).strip()
    max_logit = first_token_scores[0, first_token_id].item()

    # Get top-5 tokens from first position scores
    top5_values, top5_ids = first_token_scores[0].topk(5)
    top5_ids_list = top5_ids.tolist()
    top5_tokens = [
        processor.tokenizer.decode([tid], skip_special_tokens=True).strip()
        for tid in top5_ids_list
    ]
    top5_logits = top5_values.tolist()

    # Get GT token IDs
    gt_token_ids = processor.tokenizer.encode(ground_truth, add_special_tokens=False)

    # Check correctness by comparing token IDs
    if len(gt_token_ids) == 1:
        # Single token GT (e.g., "left", "right"): exact match
        correct = first_token_id == gt_token_ids[0]
    else:
        # Multi-token GT (e.g., "behind" = ["beh", "ind"]): match first token
        correct = first_token_id == gt_token_ids[0]

    # Check if GT (first token) is in top-5
    gt_first_token_id = gt_token_ids[0]
    gt_in_top5 = gt_first_token_id in top5_ids_list

    # Constrained evaluation: argmax over valid option logits
    option_logits = None
    constrained_choice = None
    correct_constrained = None
    if prepositions is not None:
        option_logits = {}
        for prep in sorted(prepositions):
            prep_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
            logit = first_token_scores[0, prep_ids[0]].item()
            option_logits[prep] = float(logit)
        constrained_choice = max(option_logits, key=option_logits.get)
        correct_constrained = constrained_choice == ground_truth

    return {
        "prediction": generated_text,
        "ground_truth": ground_truth,
        "correct": correct,
        "correct_constrained": correct_constrained,
        "constrained_choice": constrained_choice,
        "option_logits": option_logits,
        "first_token": first_token,
        "first_token_id": first_token_id,
        "gt_token_ids": gt_token_ids,
        "max_logit": float(max_logit),
        "top5_tokens": top5_tokens,
        "top5_ids": top5_ids_list,
        "top5_logits": top5_logits,
        "gt_in_top5": gt_in_top5,
    }


def run_knockout_experiment(
    dataset_path,
    dataset_name,
    model_path,
    n_samples=None,
    layer_range=None,
    knockout_type="direct",
    knockout_fraction=1.0,
    cluster=None,
    baseline_results=None,
    complement=False,
    question_format="forced_choice",
    task_type="spatial_relative",
):
    """
    Run knockout experiment comparing baseline vs masked attention.

    Args:
        dataset_path: Base path to datasets directory
        dataset_name: Name of the dataset (e.g., "controlled_clevr")
        model_path: Path to Qwen3VL model
        n_samples: Number of samples to process (None = all)
        layer_range: Optional (start, end) for layer-specific knockout [start, end)
        knockout_type: "direct" (last→image), "mediated" (text→image), or "cluster"
        knockout_fraction: Fraction of image tokens to block (0.0 to 1.0)
        cluster: Cluster name for knockout ("object1", "object2", "relation", "format")
        baseline_results: Pre-loaded baseline results (skips baseline forward pass)

    Returns:
        Results dict with baseline and knockout accuracy
    """
    # Load model and dataset using shared utilities
    model, processor = load_model(model_path)
    dataset, prepositions = load_dataset(dataset_path, dataset_name)

    # Determine how many samples to process
    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))

    layer_str = (
        f"layers {layer_range[0]}-{layer_range[1]}" if layer_range else "all layers"
    )
    if knockout_type == "direct":
        knockout_desc = "last→image (direct)"
    elif knockout_type == "mediated":
        knockout_desc = "text→image (mediated)"
    elif knockout_type == "cluster":
        if complement:
            knockout_desc = f"COMPLEMENT: only {cluster} sees image (block rest→image)"
        else:
            knockout_desc = f"cluster→image ({cluster})"
    elif knockout_type == "last_to_text":
        knockout_desc = "last→text (readout)"
    elif knockout_type == "last_to_cluster":
        knockout_desc = f"last→cluster ({cluster})"
    else:
        knockout_desc = knockout_type
    fraction_str = (
        f"{knockout_fraction*100:.0f}%" if knockout_fraction < 1.0 else "100%"
    )
    if knockout_type in ("cluster", "last_to_text", "last_to_cluster"):
        print(
            f"\nProcessing {n_samples} samples with {knockout_desc} knockout on {layer_str}..."
        )
    else:
        print(
            f"\nProcessing {n_samples} samples with {knockout_desc} knockout ({fraction_str} of image tokens) on {layer_str}..."
        )

    # Use preloaded baseline or run it
    has_baseline = baseline_results is not None
    if has_baseline:
        # Limit baseline to n_samples and build index
        baseline_results = [r for r in baseline_results if r["sample_idx"] < n_samples]
        baseline_by_idx = {r["sample_idx"]: r for r in baseline_results}
        print(f"Using preloaded baseline ({len(baseline_results)} samples)")
    else:
        baseline_results = []

    knockout_results = []

    for idx in tqdm(range(n_samples), desc="Evaluating samples"):
        try:
            sample = dataset[idx]
            image = sample["image"]
            objects = sample["objects"]
            ground_truth = sample["preposition"]

            # Use per-sample question_text if available (e.g., coco_recognition_pairs),
            # otherwise generate from task_type
            if "question_text" in sample and sample["question_text"]:
                question = sample["question_text"]
                if question_format == "open":
                    question += " Answer with a single word."
            else:
                question = create_question(objects, prepositions, question_format=question_format,
                                           task_type=task_type)

            # Baseline (skip if preloaded)
            if not has_baseline:
                baseline_result = evaluate_sample(
                    model, processor, image, question, ground_truth, use_knockout=False,
                    prepositions=prepositions,
                )
                baseline_result["sample_idx"] = idx
                baseline_results.append(baseline_result)

            # Knockout
            knockout_result = evaluate_sample(
                model,
                processor,
                image,
                question,
                ground_truth,
                use_knockout=True,
                knockout_type=knockout_type,
                knockout_fraction=knockout_fraction,
                layer_range=layer_range,
                objects=objects,
                cluster=cluster,
                prepositions=prepositions,
                complement=complement,
                verbose=(idx == 0),
            )
            knockout_result["sample_idx"] = idx
            knockout_results.append(knockout_result)

            # Free memory
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Compute accuracy (first token ID match)
    baseline_acc = np.mean([r["correct"] for r in baseline_results]) * 100
    knockout_acc = np.mean([r["correct"] for r in knockout_results]) * 100
    acc_drop = knockout_acc - baseline_acc

    # Also compute GT in top-5 (for analysis)
    baseline_top5 = np.mean([r["gt_in_top5"] for r in baseline_results]) * 100
    knockout_top5 = np.mean([r["gt_in_top5"] for r in knockout_results]) * 100

    # Constrained accuracy (argmax over valid options)
    has_constrained = any(r.get("correct_constrained") is not None for r in knockout_results)
    if has_constrained:
        baseline_acc_c = np.mean([r.get("correct_constrained", False) for r in baseline_results]) * 100
        knockout_acc_c = np.mean([r.get("correct_constrained", False) for r in knockout_results]) * 100
        acc_drop_c = knockout_acc_c - baseline_acc_c
    else:
        baseline_acc_c = None
        knockout_acc_c = None
        acc_drop_c = None

    results = {
        # Main accuracy (first token ID match)
        "baseline_accuracy": float(baseline_acc),
        "knockout_accuracy": float(knockout_acc),
        "accuracy_drop": float(acc_drop),
        # Constrained accuracy (argmax over valid options)
        "baseline_accuracy_constrained": float(baseline_acc_c) if baseline_acc_c is not None else None,
        "knockout_accuracy_constrained": float(knockout_acc_c) if knockout_acc_c is not None else None,
        "accuracy_drop_constrained": float(acc_drop_c) if acc_drop_c is not None else None,
        # GT in top-5 (for analysis)
        "baseline_top5": float(baseline_top5),
        "knockout_top5": float(knockout_top5),
        # Metadata
        "n_samples": len(baseline_results),
        "layer_range": layer_range,
        "knockout_type": knockout_type,
        "knockout_fraction": knockout_fraction,
        "cluster": cluster,
        "question_format": question_format,
        "baseline_results": baseline_results,
        "knockout_results": knockout_results,
    }

    return results


def print_results(results):
    """Pretty print experiment results."""
    knockout_type = results.get("knockout_type", "direct")
    knockout_fraction = results.get("knockout_fraction", 1.0)
    cluster = results.get("cluster")

    print("\n" + "=" * 70)
    if knockout_type == "direct":
        print("EXPERIMENT 02.1: KNOCKOUT DIRECT PATH (Last → Image)")
    elif knockout_type == "mediated":
        print("EXPERIMENT 02.2: KNOCKOUT MEDIATED PATH (Text → Image)")
    elif knockout_type == "cluster":
        print(f"EXPERIMENT 02.3: KNOCKOUT CLUSTER PATH ({cluster} → Image)")
    elif knockout_type == "last_to_text":
        print("EXPERIMENT 02.4: KNOCKOUT READOUT PATH (Last → Text)")
    elif knockout_type == "last_to_cluster":
        print(f"EXPERIMENT 02.5: KNOCKOUT READOUT CLUSTER (Last → {cluster})")
    print("=" * 70)

    if results["layer_range"]:
        print(
            f"Knockout applied to: Layers {results['layer_range'][0]}-{results['layer_range'][1]-1}"
        )
    else:
        print("Knockout applied to: All layers")

    if knockout_type == "cluster":
        print(f"Cluster knocked out: {cluster}")
    else:
        fraction_str = f"{knockout_fraction*100:.0f}%"
        print(f"Image tokens knocked out: {fraction_str}")
    print(f"Samples evaluated: {results['n_samples']}")
    print()

    # Accuracy comparison
    qfmt = results.get("question_format", "forced_choice")
    has_constrained = results.get("baseline_accuracy_constrained") is not None

    if qfmt == "open" and has_constrained:
        print("ACCURACY (constrained — argmax over valid options):")
        print(f"                  Baseline    Knockout    Drop")
        print(
            f"  Accuracy:       {results['baseline_accuracy_constrained']:5.1f}%      {results['knockout_accuracy_constrained']:5.1f}%    {results['accuracy_drop_constrained']:+5.1f}%"
        )
        print()
        print(f"  (also stored: top-1 full vocab baseline={results['baseline_accuracy']:.1f}%, knockout={results['knockout_accuracy']:.1f}%)")
    else:
        print("ACCURACY (first token ID match):")
        print(f"                  Baseline    Knockout    Drop")
        print(
            f"  Accuracy:       {results['baseline_accuracy']:5.1f}%      {results['knockout_accuracy']:5.1f}%    {results['accuracy_drop']:+5.1f}%"
        )
        if has_constrained:
            print()
            print("ACCURACY (constrained — argmax over valid options):")
            print(f"                  Baseline    Knockout    Drop")
            print(
                f"  Accuracy:       {results['baseline_accuracy_constrained']:5.1f}%      {results['knockout_accuracy_constrained']:5.1f}%    {results['accuracy_drop_constrained']:+5.1f}%"
            )
    print()
    print(
        f"  GT in top-5: Baseline {results['baseline_top5']:.1f}%, Knockout {results['knockout_top5']:.1f}%"
    )
    print()

    # Interpretation
    acc_drop = results["accuracy_drop"]


def plot_accuracy_comparison(results, output_dir):
    """Create bar plot comparing baseline vs knockout accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data for bar chart
    conditions = ["Baseline", "Knockout"]
    values = [results["baseline_accuracy"], results["knockout_accuracy"]]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(conditions, values, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add drop annotation
    drop = results["accuracy_drop"]
    mid_height = (values[0] + values[1]) / 2
    ax.annotate(
        f"Drop: {drop:+.1f}%",
        xy=(0.5, mid_height),
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    knockout_type = results.get("knockout_type", "direct")
    cluster = results.get("cluster")
    if knockout_type == "direct":
        title = (
            "Experiment 02.1: Direct Path Knockout\n(Last → Image Attention Blocked)"
        )
    elif knockout_type == "mediated":
        title = (
            "Experiment 02.2: Mediated Path Knockout\n(Text → Image Attention Blocked)"
        )
    elif knockout_type == "cluster":
        title = (
            f"Experiment 02.3: Cluster Knockout\n({cluster} → Image Attention Blocked)"
        )
    elif knockout_type == "last_to_text":
        title = (
            "Experiment 02.4: Readout Knockout\n(Last → Text Attention Blocked)"
        )
    elif knockout_type == "last_to_cluster":
        title = (
            f"Experiment 02.5: Readout Cluster Knockout\n(Last → {cluster} Attention Blocked)"
        )
    else:
        title = f"Knockout Experiment ({knockout_type})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to {output_path}")
    plt.close()


def save_results(results, output_dir, dataset_name=None):
    """Save results to JSON and create visualizations."""
    if dataset_name:
        output_dir = output_dir / dataset_name

    # Add knockout type, fraction, and layer info to output path
    knockout_type = results.get("knockout_type", "direct")
    knockout_fraction = results.get("knockout_fraction", 1.0)
    layer_range = results.get("layer_range")
    cluster = results.get("cluster")

    question_format = results.get("question_format", "forced_choice")

    # Map knockout type to readable directory name
    KO_DIR_NAMES = {
        "direct": "last_to_image",
        "mediated": "text_to_image",
        "last_to_text": "last_to_text",
    }

    if knockout_type == "cluster":
        subdir = f"cluster_{cluster}"
    elif knockout_type == "last_to_cluster":
        subdir = f"last_to_cluster_{cluster}"
    elif knockout_type in KO_DIR_NAMES:
        subdir = KO_DIR_NAMES[knockout_type]
    else:
        subdir = f"{knockout_type}_frac{knockout_fraction:.2f}"
    if layer_range:
        subdir += f"_layers{layer_range[0]}-{layer_range[1]}"
    # question_format distinction is handled by the top-level output dir
    # (knockout/ vs knockout_open/), so no suffix needed here

    output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary (without full results to keep file small)
    summary = {
        "baseline_accuracy": results["baseline_accuracy"],
        "knockout_accuracy": results["knockout_accuracy"],
        "accuracy_drop": results["accuracy_drop"],
        "baseline_accuracy_constrained": results.get("baseline_accuracy_constrained"),
        "knockout_accuracy_constrained": results.get("knockout_accuracy_constrained"),
        "accuracy_drop_constrained": results.get("accuracy_drop_constrained"),
        "baseline_top5": results["baseline_top5"],
        "knockout_top5": results["knockout_top5"],
        "n_samples": results["n_samples"],
        "layer_range": results["layer_range"],
        "knockout_type": results.get("knockout_type", "direct"),
        "knockout_fraction": results.get("knockout_fraction", 1.0),
        "cluster": results.get("cluster"),
        "question_format": results.get("question_format", "forced_choice"),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {output_dir / 'summary.json'}")

    # Save detailed results
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to {output_dir / 'detailed_results.json'}")

    # Create visualization
    plot_accuracy_comparison(results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 02: Attention knockout experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--dataset-path", type=str, help="Base path to datasets")
    parser.add_argument("--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument("--model-path", type=str, help="Path to Qwen3-VL model")
    parser.add_argument("--n-samples", type=int, help="Number of samples to process")
    parser.add_argument("--layers", type=str, help="Layer range for knockout")
    parser.add_argument(
        "--knockout-type",
        type=str,
        choices=["direct", "mediated", "cluster", "last_to_text", "last_to_cluster"],
        default=None,
        help="Knockout type: 'direct' (last→image), 'mediated' (text→image), 'cluster' (cluster→image), "
        "'last_to_text' (last→text), or 'last_to_cluster' (last→cluster)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Cluster to knock out. Use '+' to merge (e.g., object1+object2). "
        "Available: object1, object2, relation, format, where, in_relation, "
        "question_mark, answer_with, correct_prep, wrong_preps, punctuation",
    )
    parser.add_argument(
        "--complement",
        action="store_true",
        help="Invert: block everything EXCEPT the named cluster from seeing image",
    )
    parser.add_argument(
        "--knockout-fraction",
        type=float,
        default=None,
        help="Fraction of image tokens to block (0.0 to 1.0, default 1.0)",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Path to baseline diagnostic_samples.json (skips baseline forward pass)",
    )
    parser.add_argument(
        "--question-format",
        type=str,
        default="forced_choice",
        choices=["forced_choice", "open", "free"],
        help="Question format: forced_choice (with alternatives), "
             "open (single word, no alternatives), free (bare question).",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="spatial_relative",
        choices=["spatial_relative", "spatial_absolute", "recognition"],
        help="Task type: spatial_relative (2 objects, default), "
             "spatial_absolute (1 object, absolute position), "
             "recognition (what object is in image).",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=None,
        help="Override results subdirectory (e.g., 'qwen3/knockout_open').",
    )

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)

    # Override with command-line args
    if args.dataset_path is not None:
        config["dataset_path"] = args.dataset_path
    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    if args.model_path is not None:
        config["model_path"] = args.model_path
    if args.n_samples is not None:
        config["n_samples"] = args.n_samples
    if args.layers is not None:
        config["layers"] = args.layers
    if args.knockout_type is not None:
        config["knockout_type"] = args.knockout_type
    if args.cluster is not None:
        config["cluster"] = args.cluster
    if args.knockout_fraction is not None:
        config["knockout_fraction"] = args.knockout_fraction
    if args.no_save:
        config["save"] = False

    # Parse layer range
    layer_range = parse_layer_range(config.get("layers"))

    # Set defaults
    config.setdefault("save", True)
    config.setdefault("n_samples", None)
    config.setdefault("knockout_type", "direct")
    config.setdefault("knockout_fraction", 1.0)
    config.setdefault("cluster", None)

    # Validate required args
    required = ["dataset_path", "dataset_name", "model_path"]
    missing = [k for k in required if k not in config]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    # Validate cluster knockout
    if config["knockout_type"] in ("cluster", "last_to_cluster") and config["cluster"] is None:
        parser.error("--cluster is required when --knockout-type is 'cluster' or 'last_to_cluster'")

    # Load baseline if provided
    baseline_results = None
    if args.baseline_results:
        baseline_results = load_baseline_results(args.baseline_results)
        print(
            f"Loaded {len(baseline_results)} baseline results from {args.baseline_results}"
        )

    # Run experiment
    results = run_knockout_experiment(
        dataset_path=config["dataset_path"],
        dataset_name=config["dataset_name"],
        model_path=config["model_path"],
        n_samples=config["n_samples"],
        layer_range=layer_range,
        knockout_type=config["knockout_type"],
        knockout_fraction=config["knockout_fraction"],
        cluster=config["cluster"],
        baseline_results=baseline_results,
        complement=args.complement,
        question_format=args.question_format,
        task_type=args.task_type,
    )

    # Print results
    print_results(results)

    # Save results
    if config["save"]:
        if args.results_subdir:
            subdir = args.results_subdir
        else:
            subdir = "knockout" if args.question_format == "forced_choice" else f"knockout_{args.question_format}"
        output_dir = Path(__file__).parent.parent / "results" / subdir
        save_results(results, output_dir, dataset_name=config["dataset_name"])


if __name__ == "__main__":
    main()
