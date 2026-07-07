"""
    python experiments/causal_tracing.py \
        --config configs/default.yaml \
        --dataset-path data/ \
        --dataset-name controlled_shapes_center \
        --n-pairs 20 \
        --append-to results/causal_tracing/results.json \
        --groups "object1+in_relation+object2,all_text-object1-in_relation-object2"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from vlm_spatial import (
    load_model,
    load_dataset,
    create_question,
    find_token_ranges,
    find_cluster_ranges,
    load_config,
    get_language_layers,
)


def _extract_hidden_state(output):
    """Extract hidden state tensor from decoder layer output.

    Handles both tuple output (hidden_states, ...) and direct tensor output.
    Always returns a 3D tensor [batch, seq_len, hidden_dim].
    """
    if isinstance(output, tuple):
        hs = output[0]
    else:
        hs = output
    # Ensure 3D
    if hs.dim() == 2:
        hs = hs.unsqueeze(0)
    return hs


def _replace_hidden_state(output, new_hs):
    """Replace hidden state in decoder layer output, preserving format."""
    if isinstance(output, tuple):
        # If original was 2D inside tuple, squeeze back
        if output[0].dim() == 2:
            new_hs = new_hs.squeeze(0)
        return (new_hs,) + output[1:]
    else:
        if output.dim() == 2:
            new_hs = new_hs.squeeze(0)
        return new_hs


def _get_logits(model, inputs):
    """Single forward pass, return logits at last position [vocab_size]."""
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.logits: [batch, seq_len, vocab_size]
    return outputs.logits[0, -1, :]


def collect_hidden_states(model, inputs, layers_to_collect):
    """Run single forward pass and collect hidden states at specified layers.

    Args:
        model: Qwen3VL model
        inputs: Processor outputs (on device)
        layers_to_collect: List of layer indices to collect states from

    Returns:
        Tuple of (hidden_states_dict, logits_at_last_pos)
        hidden_states_dict maps layer_idx -> tensor [1, seq_len, hidden_dim]
    """
    hidden_states = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = _extract_hidden_state(output)
            hidden_states[layer_idx] = hs.detach().clone()

        return hook_fn

    lm_layers = get_language_layers(model)
    for layer_idx in layers_to_collect:
        handle = lm_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    logits = outputs.logits[0, -1, :]  # [vocab_size]
    return hidden_states, logits


def run_with_patch(
    model, inputs, clean_states, patch_layer, patch_token_indices, gt_token_id,
    source_layer=None, source_token_indices=None, option_token_ids=None,
):
    """Run corrupted forward pass but restore clean hidden states for specific tokens at a layer.

    Args:
        model: Qwen3VL model
        inputs: Corrupted inputs (on device)
        clean_states: Dict of clean hidden states {layer_idx: tensor [1, seq, hidden]}
        patch_layer: Which layer to patch at (target)
        patch_token_indices: List of token indices to write TO in the corrupted run
        gt_token_id: Token ID of the correct answer (to measure probability)
        source_layer: Which layer's clean states to use (default: same as patch_layer).
                      Set differently for cross-layer patching.
        source_token_indices: List of token indices to read FROM in clean states.
                              Default: same as patch_token_indices (standard same-position patching).
                              Set differently for cross-position patching (e.g. object moves between images).

    Returns:
        P(correct_answer) after patching
    """
    if source_layer is None:
        source_layer = patch_layer
    if source_token_indices is None:
        source_token_indices = patch_token_indices

    hooks = []

    def patch_hook(module, input, output):
        hs = _extract_hidden_state(output)  # [1, seq_len, hidden_dim]
        patched = hs.clone()
        clean = clean_states[source_layer]  # [1, seq_len, hidden_dim]
        for src_idx, tgt_idx in zip(source_token_indices, patch_token_indices):
            patched[0, tgt_idx, :] = clean[0, src_idx, :]
        return _replace_hidden_state(output, patched)

    lm_layers = get_language_layers(model)
    handle = lm_layers[patch_layer].register_forward_hook(patch_hook)
    hooks.append(handle)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    # Get probability of correct answer and argmax prediction
    logits = outputs.logits[0, -1, :]  # [vocab_size]
    probs = torch.softmax(logits, dim=-1)
    p_gt = probs[gt_token_id].item()
    argmax_id = logits.argmax().item()

    # Optionally return logits for all option tokens
    if option_token_ids is not None:
        option_logits = {name: logits[tid].item()
                        for name, tid in option_token_ids.items()}
        return p_gt, argmax_id, option_logits
    return p_gt, argmax_id


def get_answer_prob(model, inputs, gt_token_id):
    """Run forward pass and return P(correct_answer) and argmax token ID."""
    logits = _get_logits(model, inputs)
    probs = torch.softmax(logits, dim=-1)
    return probs[gt_token_id].item(), logits.argmax().item()


def prepare_inputs(processor, image, question):
    """Prepare model inputs from image and question."""
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
    )
    return inputs


def _get_image_grid(inputs, n_image_tokens):
    """Get spatial grid dimensions (h, w) of image tokens.

    Uses image_grid_thw from processor if available, otherwise infers from token count.
    """
    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
        thw = inputs.image_grid_thw[0].tolist()
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        if h * w == n_image_tokens:
            return h, w

    # Fallback: assume square grid
    side = int(n_image_tokens**0.5)
    if side * side == n_image_tokens:
        return side, side

    # Cannot determine grid — return None (patch-level tracing will be skipped)
    return None


def compute_image_patch_regions(image_range, grid_hw, image_size, positions, shape_size,
                                padding=0):
    """Map image tokens to object bounding-box regions.

    Args:
        image_range: (start, end) token indices for image tokens in sequence.
        grid_hw: (grid_h, grid_w) spatial layout of image tokens.
        image_size: Image dimensions in pixels (assuming square).
        positions: List of (x, y) pixel centers for each object.
        shape_size: Object diameter in pixels.
        padding: Number of extra patches to add around the bounding box (0, 1, or 2).

    Returns:
        List of token index lists, one per object position.
    """
    img_start, img_end = image_range
    grid_h, grid_w = grid_hw

    cell_h = image_size / grid_h
    cell_w = image_size / grid_w
    obj_r = shape_size / 2

    results = []
    for cx, cy in positions:
        left = cx - obj_r
        right = cx + obj_r
        top = cy - obj_r
        bottom = cy + obj_r

        # Find patch grid rows/cols that overlap the bounding box
        col_start = int(left // cell_w)
        col_end = int(np.ceil(right / cell_w))
        row_start = int(top // cell_h)
        row_end = int(np.ceil(bottom / cell_h))

        # Apply padding
        col_start = max(0, col_start - padding)
        col_end = min(grid_w, col_end + padding)
        row_start = max(0, row_start - padding)
        row_end = min(grid_h, row_end + padding)

        indices = []
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                token_idx = img_start + r * grid_w + c
                indices.append(token_idx)

        results.append(indices)

    return results


def _resolve_cluster_indices(cluster_ranges, name):
    """Resolve a cluster name to a list of token indices."""
    spec = cluster_ranges.get(name)
    if spec is None:
        return []
    if isinstance(spec, tuple):
        return list(range(spec[0], spec[1]))
    # List of tuples
    indices = []
    for start, end in spec:
        indices.extend(range(start, end))
    return indices


# Cluster names in prompt order
CLUSTER_NAMES_ORDERED = [
    "where",
    "object1",
    "in_relation",
    "object2",
    "question_mark",
    "answer_with",
    "correct_prep",
    "wrong_preps",
    "body_correct_prep",
    "body_wrong_preps",
    "body_relator",
    "punctuation",
]


def parse_group_spec(spec, cluster_indices, all_text_indices):
    """Parse a group specification string into (name, token_indices).

    Supports:
        - Single cluster: "object1"
        - Merged clusters: "object1+in_relation+object2"
        - Complement (all_text minus clusters): "all_text-object1-in_relation-object2"

    Args:
        spec: Group specification string.
        cluster_indices: Dict of cluster_name -> list of token indices.
        all_text_indices: Set of all text token indices.

    Returns:
        Tuple of (group_name, sorted list of token indices).
    """
    if spec.startswith("all_text-"):
        # Complement: all_text minus named clusters
        to_remove = spec[len("all_text-"):].split("-")
        remaining = set(all_text_indices)
        for name in to_remove:
            if name not in cluster_indices:
                raise ValueError(f"Unknown cluster '{name}' in spec '{spec}'")
            remaining -= set(cluster_indices[name])
        return (spec, sorted(remaining))
    elif "+" in spec:
        # Merged clusters
        parts = spec.split("+")
        merged = []
        for name in parts:
            if name not in cluster_indices:
                raise ValueError(f"Unknown cluster '{name}' in spec '{spec}'")
            merged.extend(cluster_indices[name])
        return (spec, sorted(set(merged)))
    else:
        # Single cluster
        if spec not in cluster_indices:
            raise ValueError(f"Unknown cluster '{spec}'")
        return (spec, cluster_indices[spec])


def define_token_groups(ranges, cluster_ranges, combos=False, group_specs=None):
    """Define token groups for patching.

    Args:
        ranges: Token ranges from find_token_ranges.
        cluster_ranges: Cluster ranges from find_cluster_ranges.
        combos: If True, add all pairwise combinations and leave-one-out groups.
        group_specs: If provided, list of group spec strings to compute instead of
                     the default set. Supports "name+name" merges and
                     "all_text-name-name" complements.

    Returns:
        List of (group_name, token_indices) tuples.
    """
    # Resolve all cluster indices (in prompt order)
    cluster_indices = {}
    for name in CLUSTER_NAMES_ORDERED:
        indices = _resolve_cluster_indices(cluster_ranges, name)
        if indices:
            cluster_indices[name] = indices

    # All text token set (for complements)
    all_text_indices = set()
    if ranges["text"] is not None:
        txt_start, txt_end = ranges["text"]
        all_text_indices = set(range(txt_start, txt_end))

    # If specific groups requested, only compute those
    if group_specs is not None:
        groups = []
        for spec in group_specs:
            # Handle built-in names that aren't cluster-based
            if spec == "image" and ranges["image"] is not None:
                img_start, img_end = ranges["image"]
                groups.append(("image", list(range(img_start, img_end))))
            elif spec == "all_text" and all_text_indices:
                groups.append(("all_text", sorted(all_text_indices)))
            elif spec == "last_token" and ranges["last"] is not None:
                groups.append(("last_token", [ranges["last"]]))
            elif spec == "prompt_start" and ranges["prompt_start"] is not None:
                ps_start, ps_end = ranges["prompt_start"]
                groups.append(("prompt_start", list(range(ps_start, ps_end))))
            elif spec == "prompt_end" and ranges["prompt_end"] is not None:
                pe_start, pe_end = ranges["prompt_end"]
                groups.append(("prompt_end", list(range(pe_start, pe_end))))
            elif spec == "prompt_all":
                indices = []
                if ranges["prompt_start"] is not None:
                    ps_start, ps_end = ranges["prompt_start"]
                    indices.extend(range(ps_start, ps_end))
                if ranges["prompt_end"] is not None:
                    pe_start, pe_end = ranges["prompt_end"]
                    indices.extend(range(pe_start, pe_end))
                groups.append(("prompt_all", sorted(indices)))
            else:
                groups.append(parse_group_spec(spec, cluster_indices, all_text_indices))
        return groups

    groups = []

    # Image tokens (all)
    if ranges["image"] is not None:
        img_start, img_end = ranges["image"]
        groups.append(("image", list(range(img_start, img_end))))

    # Individual clusters
    for name in CLUSTER_NAMES_ORDERED:
        if name in cluster_indices:
            groups.append((name, cluster_indices[name]))

    if combos:
        available = [n for n in CLUSTER_NAMES_ORDERED if n in cluster_indices]

        # All pairwise combinations (bottom-up: which pair restores signal?)
        from itertools import combinations
        for a, b in combinations(available, 2):
            merged = cluster_indices[a] + cluster_indices[b]
            groups.append((f"{a}+{b}", sorted(merged)))

        # Leave-one-out from all_text (top-down: which cluster is necessary?)
        if ranges["text"] is not None:
            for name in available:
                remaining = sorted(all_text_indices - set(cluster_indices[name]))
                groups.append((f"all_text-{name}", remaining))
    else:
        # Just the obj1+obj2 merged group (backwards compatible)
        if "object1" in cluster_indices and "object2" in cluster_indices:
            groups.append((
                "object1+object2",
                cluster_indices["object1"] + cluster_indices["object2"],
            ))

    # All text tokens
    if ranges["text"] is not None:
        groups.append(("all_text", sorted(all_text_indices)))

    # Last token
    if ranges["last"] is not None:
        last_pos = ranges["last"]
        groups.append(("last_token", [last_pos]))

    return groups


def run_causal_tracing(
    dataset_path,
    dataset_name,
    model_path,
    n_pairs=50,
    layer_step=1,
    shape_size=None,
    image_size=448,
    combos=False,
    group_specs=None,
    question_format="forced_choice",
    patch_groups=False,
    both_patch_groups=False,
    random_patch_groups=False,
    task_type="spatial_relative",
    perspective=None,
    save_option_logits=False,
    question_override=None,
):
    """Run causal tracing experiment on paired dataset.

    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of paired dataset
        model_path: Path to model
        n_pairs: Number of pairs to process
        layer_step: Process every N-th layer (1=all, 2=every other, 4=every 4th)
        shape_size: Object diameter in pixels (for region-based patching). If None,
                    reads from dataset or skips region-based groups.
        image_size: Image size in pixels (for region-based patching).
        combos: If True, add all pairwise combinations and leave-one-out groups.
        group_specs: If provided, only compute these specific groups. Overrides combos.
        question_format: Question format for create_question.
        patch_groups: If True, add cross-position patch groups (obj1_cross,
                      obj1_cross_pad1, obj1_cross_pad2, obj2_same).
        both_patch_groups: If True, add merged obj1+obj2 patch groups (both_obj,
                          both_obj_pad1, both_obj_pad2). Requires both obj positions.
        random_patch_groups: If True, add obj1 + random non-object image token groups
                            (obj1_cross_rpad1, obj1_cross_rpad2). Tests whether specific
                            surrounding patches matter or just extra image token count.

    Returns:
        Results dict with heatmap data
    """
    model, processor = load_model(model_path)
    dataset, prepositions = load_dataset(dataset_path, dataset_name, task_type=task_type,
                                         perspective=perspective)

    # Determine number of layers
    n_layers = len(get_language_layers(model))
    layers_to_trace = list(range(0, n_layers, layer_step))
    print(
        f"Tracing {len(layers_to_trace)} layers (step={layer_step}): {layers_to_trace[0]}-{layers_to_trace[-1]}"
    )

    # Check for region-based patching support
    has_obj2_positions = "obj2_position" in dataset.column_names
    has_positions = (
        "obj1_position" in dataset.column_names
        and has_obj2_positions
    )
    has_obj1_positions = "obj1_position" in dataset.column_names
    if has_positions or has_obj1_positions:
        # Read shape_size/image_size from dataset if not provided
        if shape_size is None and "shape_size" in dataset.column_names:
            shape_size = dataset[0]["shape_size"]
        if "image_size" in dataset.column_names:
            image_size = dataset[0]["image_size"]
        if shape_size is None:
            print("Warning: dataset has positions but no shape_size. Skipping region patching.")
            has_positions = False
            has_obj1_positions = False
        else:
            print(f"Region-based patching enabled (shape_size={shape_size}, image_size={image_size})")
    else:
        print("No position data in dataset. Skipping region-based patching.")

    # Determine pairs
    if "pair_id" not in dataset.column_names:
        raise ValueError(
            "Dataset must have 'pair_id' column. Generate with --pairs flag."
        )

    max_pair_id = max(dataset["pair_id"])
    total_pairs = max_pair_id + 1
    n_pairs = min(n_pairs, total_pairs)
    print(f"Processing {n_pairs} pairs out of {total_pairs}")

    # Build pair index: pair_id -> [idx_a, idx_b]
    pair_index = {}
    for i in range(len(dataset)):
        pid = dataset[i]["pair_id"]
        if pid not in pair_index:
            pair_index[pid] = []
        pair_index[pid].append(i)

    # Accumulate results: group_name -> layer_idx -> list of restoration scores
    all_scores = {}
    # Accumulate argmax token IDs: group_name -> layer_idx -> list of argmax ids
    all_argmax = {}
    # Per-pair scores: pair_index -> group_name -> layer_idx -> score
    per_pair_scores = []
    # Per-pair metadata for argmax classification
    pair_gt_ids = []       # gt_token_id per pair (clean answer)
    pair_corrupted_ids = [] # corrupted argmax per pair
    pair_metadata = []     # per-pair: clean/corrupted tokens + probs
    group_names = None
    n_skipped = 0

    # Option logits tracking (when --save-option-logits is set)
    option_tid_map = None  # {option_name: token_id}
    all_option_scores = {}  # group -> layer -> option -> [scores...]
    if save_option_logits and prepositions:
        option_tid_map = {}
        for prep in sorted(prepositions):
            prep_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
            option_tid_map[prep] = prep_ids[0]
        print(f"Tracking option logits for: {list(option_tid_map.keys())}")

    for pair_id in tqdm(range(n_pairs), desc="Causal tracing"):
        indices = pair_index[pair_id]
        if len(indices) != 2:
            print(f"Warning: pair {pair_id} has {len(indices)} samples, skipping")
            continue

        idx_a, idx_b = indices[0], indices[1]
        sample_a = dataset[idx_a]
        sample_b = dataset[idx_b]

        # Same objects for both
        objects = sample_a["objects"]
        # Question: override > per-sample question_text > generated from task_type
        if question_override:
            question = question_override
            if question_format == "open":
                question += " Answer with a single word."
            elif question_format == "forced_choice":
                question += f" Answer only with {', '.join(sorted(prepositions))}."
        elif "question_text" in sample_a and sample_a["question_text"]:
            question = sample_a["question_text"]
            if question_format == "open":
                question += " Answer with a single word."
        else:
            question = create_question(objects, prepositions, question_format=question_format,
                                       task_type=task_type, perspective=perspective)
        gt_a = sample_a["preposition"]

        # Encode ground truth token ID
        gt_token_ids = processor.tokenizer.encode(gt_a, add_special_tokens=False)
        gt_token_id = gt_token_ids[0]

        # Prepare inputs
        inputs_a = prepare_inputs(processor, sample_a["image"], question)
        inputs_a = inputs_a.to(model.device)
        inputs_b = prepare_inputs(processor, sample_b["image"], question)
        inputs_b = inputs_b.to(model.device)

        # Get token ranges and cluster ranges (from clean input)
        ranges = find_token_ranges(inputs_a.input_ids, processor.tokenizer)
        cluster_ranges = find_cluster_ranges(
            inputs_a.input_ids,
            processor.tokenizer,
            objects,
            ground_truth=gt_a,
            prepositions=prepositions,
        )

        # Define token groups (only once for names, but resolve indices each time)
        token_groups = define_token_groups(
            ranges, cluster_ranges, combos=combos, group_specs=group_specs,
        )

        # Resolve last token index (-1 → absolute position)
        seq_len = inputs_a.input_ids.shape[1]
        token_groups = [
            (name, [idx if idx >= 0 else seq_len + idx for idx in indices])
            for name, indices in token_groups
        ]

        # Compute B's token groups when sequence lengths differ (e.g. different images)
        seq_len_b = inputs_b.input_ids.shape[1]
        if seq_len_b != seq_len:
            ranges_b = find_token_ranges(inputs_b.input_ids, processor.tokenizer)
            cluster_ranges_b = find_cluster_ranges(
                inputs_b.input_ids,
                processor.tokenizer,
                objects,
                ground_truth=sample_b["preposition"],
                prepositions=prepositions,
            )
            token_groups_b = define_token_groups(
                ranges_b, cluster_ranges_b, combos=combos, group_specs=group_specs,
            )
            token_groups_b = [
                (name, [idx if idx >= 0 else seq_len_b + idx for idx in indices])
                for name, indices in token_groups_b
            ]
        else:
            token_groups_b = token_groups

        # Add region-based groups if position data available (skip for custom groups)
        if has_positions and ranges["image"] is not None and group_specs is None:
            obj1_pos = tuple(sample_a["obj1_position"])
            obj2_pos = tuple(sample_a["obj2_position"])
            n_img_tokens = ranges["image"][1] - ranges["image"][0]
            grid_hw = _get_image_grid(inputs_a, n_img_tokens)
            if grid_hw is None:
                if pair_id == 0:
                    print(f"  Skipping patch-level tracing (cannot determine grid for {n_img_tokens} image tokens)")
                has_positions = False  # disable for remaining pairs
            else:
                regions = compute_image_patch_regions(
                    image_range=ranges["image"],
                    grid_hw=grid_hw,
                    image_size=image_size,
                    positions=[obj1_pos, obj2_pos],
                    shape_size=shape_size,
                )
                obj1_patches, obj2_patches = regions
                token_groups.append(("center_obj_patches", obj2_patches))
                token_groups.append(("peripheral_obj_patches", obj1_patches))

        if group_names is None:
            group_names = [g[0] for g in token_groups]
            for name in group_names:
                all_scores[name] = {l: [] for l in layers_to_trace}
                all_argmax[name] = {l: [] for l in layers_to_trace}

        # 1. Clean run: collect hidden states and P(correct)
        # Always collect layer 0 for cross-layer patching
        layers_to_collect = sorted(set(layers_to_trace) | {0})
        clean_states, clean_logits = collect_hidden_states(
            model, inputs_a, layers_to_collect
        )
        clean_probs = torch.softmax(clean_logits, dim=-1)
        p_clean = clean_probs[gt_token_id].item()

        # Skip pairs where the model gets the clean run wrong
        # Use constrained evaluation when prepositions are available:
        # check if GT has the highest logit among valid options
        clean_argmax = clean_logits.argmax().item()
        if prepositions and len(prepositions) > 1:
            # Constrained check: argmax over valid option logits
            option_logits = {}
            for prep in prepositions:
                prep_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
                option_logits[prep] = clean_logits[prep_ids[0]].item()
            constrained_choice = max(option_logits, key=option_logits.get)
            clean_correct = constrained_choice == gt_a
        else:
            clean_correct = clean_argmax == gt_token_id

        if not clean_correct:
            if clean_argmax != gt_token_id:
                # Check text match (handles " red" vs "red" token variants)
                clean_answer = processor.tokenizer.decode([clean_argmax]).strip().lower()
                gt_answer = processor.tokenizer.decode([gt_token_id]).strip().lower()
                if clean_answer == gt_answer:
                    # Same word, different token ID — use the model's token ID
                    gt_token_id = clean_argmax
                    p_clean = clean_probs[gt_token_id].item()
                else:
                    n_skipped += 1
                    if n_skipped <= 5:
                        print(f"  Skipping pair {pair_id}: model answered '{constrained_choice if prepositions else clean_answer}' "
                              f"instead of '{gt_a}' on clean run")
                    continue
            else:
                n_skipped += 1
                continue

        # 2. Corrupted run: P(correct) with wrong image
        corrupted_logits = _get_logits(model, inputs_b)
        corrupted_probs = torch.softmax(corrupted_logits, dim=-1)
        p_corrupted = corrupted_probs[gt_token_id].item()
        corrupted_argmax = corrupted_logits.argmax().item()

        # Corrupted-side filter: model must answer correctly on corrupted image too
        # (i.e. it should predict sample_b's ground truth, not sample_a's)
        gt_b = sample_b["preposition"]
        skip_reason = None
        if prepositions and len(prepositions) > 1:
            corrupted_option_logits = {}
            for prep in prepositions:
                prep_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
                corrupted_option_logits[prep] = corrupted_logits[prep_ids[0]].item()
            corrupted_constrained = max(corrupted_option_logits, key=corrupted_option_logits.get)
            if corrupted_constrained != gt_b:
                skip_reason = f"corrupted_wrong:{corrupted_constrained}"
        else:
            corrupted_answer = processor.tokenizer.decode([corrupted_argmax]).strip().lower()
            if corrupted_answer != gt_b.lower():
                skip_reason = f"corrupted_wrong:{corrupted_answer}"

        gap = p_clean - p_corrupted

        # Always save metadata (even for skipped pairs, for notebook diagnostics)
        pair_metadata.append({
            "pair_id": pair_id,
            "gt_token_id": gt_token_id,
            "gt_token": processor.tokenizer.decode([gt_token_id]).strip(),
            "gt_label": gt_a,
            "corrupted_token_id": corrupted_argmax,
            "corrupted_token": processor.tokenizer.decode([corrupted_argmax]).strip(),
            "corrupted_label": gt_b,
            "p_clean": round(p_clean, 6),
            "p_corrupted": round(p_corrupted, 6),
            "gap": round(gap, 6),
            "skipped": skip_reason,
        })

        if skip_reason is not None:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  Skipping pair {pair_id}: {skip_reason} "
                      f"(gt_b='{gt_b}')")
            continue

        pair_gt_ids.append(gt_token_id)
        pair_corrupted_ids.append(corrupted_argmax)

        # 3. Patched runs: for each group × layer, restore clean states
        pair_idx = len(pair_gt_ids) - 1  # index into pair metadata
        pair_scores_entry = {}

        # Compute per-option baselines for this pair (reuse existing logits)
        if option_tid_map is not None:
            clean_option_logits = {opt: clean_logits[tid].item()
                                   for opt, tid in option_tid_map.items()}
            corrupt_option_logits = {opt: corrupted_logits[tid].item()
                                     for opt, tid in option_tid_map.items()}

        for (group_name, src_indices), (_, tgt_indices) in zip(token_groups, token_groups_b):
            pair_scores_entry[group_name] = {}

            # Initialize option score accumulators for this group
            if option_tid_map is not None and group_name not in all_option_scores:
                all_option_scores[group_name] = {
                    l: {opt: [] for opt in option_tid_map}
                    for l in layers_to_trace
                }

            for layer_idx in layers_to_trace:
                result = run_with_patch(
                    model,
                    inputs_b,
                    clean_states,
                    patch_layer=layer_idx,
                    patch_token_indices=tgt_indices,
                    gt_token_id=gt_token_id,
                    source_token_indices=src_indices,
                    option_token_ids=option_tid_map,
                )

                if option_tid_map is not None:
                    p_patched, patched_argmax, patched_option_logits = result
                else:
                    p_patched, patched_argmax = result

                # Restoration score: how much of the clean-corrupted gap is recovered
                if abs(gap) > 1e-6:
                    score = (p_patched - p_corrupted) / gap
                else:
                    score = 0.0

                all_scores[group_name][layer_idx].append(score)
                all_argmax[group_name][layer_idx].append(patched_argmax)
                pair_scores_entry[group_name][layer_idx] = round(score, 6)

                # Per-option logit restoration scores
                if option_tid_map is not None:
                    for opt in option_tid_map:
                        opt_gap = clean_option_logits[opt] - corrupt_option_logits[opt]
                        if abs(opt_gap) > 1e-6:
                            opt_score = (patched_option_logits[opt] - corrupt_option_logits[opt]) / opt_gap
                        else:
                            opt_score = 0.0
                        all_option_scores[group_name][layer_idx][opt].append(opt_score)

        per_pair_scores.append({"pair_id": pair_id, "scores": pair_scores_entry})

        # 4. Cross-layer patching: inject layer-0 image tokens at each layer
        #    (only when running default groups, not custom group_specs)
        img_indices_src = None
        img_indices_tgt = None
        if group_specs is None:
            for name, indices in token_groups:
                if name == "image":
                    img_indices_src = indices
                    break
            for name, indices in token_groups_b:
                if name == "image":
                    img_indices_tgt = indices
                    break

        if img_indices_src is not None and img_indices_tgt is not None:
            cross_name = "image(from_L0)"
            if cross_name not in all_scores:
                group_names.append(cross_name)
                all_scores[cross_name] = {l: [] for l in layers_to_trace}
                all_argmax[cross_name] = {l: [] for l in layers_to_trace}

            for layer_idx in layers_to_trace:
                p_patched, patched_argmax = run_with_patch(
                    model,
                    inputs_b,
                    clean_states,
                    patch_layer=layer_idx,
                    patch_token_indices=img_indices_tgt,
                    gt_token_id=gt_token_id,
                    source_layer=0,  # always use layer 0 representations
                    source_token_indices=img_indices_src,
                )

                gap = p_clean - p_corrupted
                if abs(gap) > 1e-6:
                    score = (p_patched - p_corrupted) / gap
                else:
                    score = 0.0

                all_scores[cross_name][layer_idx].append(score)
                all_argmax[cross_name][layer_idx].append(patched_argmax)

        # 5. Cross-layer patching for center object patches
        if has_positions and group_specs is None:
            center_indices = None
            for name, indices in token_groups:
                if name == "center_obj_patches":
                    center_indices = indices
                    break

            if center_indices is not None:
                cross_name = "center_obj(from_L0)"
                if cross_name not in all_scores:
                    group_names.append(cross_name)
                    all_scores[cross_name] = {l: [] for l in layers_to_trace}
                    all_argmax[cross_name] = {l: [] for l in layers_to_trace}

                for layer_idx in layers_to_trace:
                    p_patched, patched_argmax = run_with_patch(
                        model,
                        inputs_b,
                        clean_states,
                        patch_layer=layer_idx,
                        patch_token_indices=center_indices,
                        gt_token_id=gt_token_id,
                        source_layer=0,
                    )

                    gap = p_clean - p_corrupted
                    if abs(gap) > 1e-6:
                        score = (p_patched - p_corrupted) / gap
                    else:
                        score = 0.0

                    all_scores[cross_name][layer_idx].append(score)
                    all_argmax[cross_name][layer_idx].append(patched_argmax)

        # 6. Cross-position patch groups: restore obj1 from clean position to corrupted position
        if patch_groups and has_obj1_positions and ranges["image"] is not None:
            obj1_pos_a = tuple(sample_a["obj1_position"])
            obj1_pos_b = tuple(sample_b["obj1_position"])

            n_img_tokens = ranges["image"][1] - ranges["image"][0]
            grid_hw = _get_image_grid(inputs_a, n_img_tokens)

            # Compute regions at each padding level for both images
            cross_groups = []
            for pad, suffix in [(0, ""), (1, "_pad1"), (2, "_pad2")]:
                # Source: obj1 patches in clean image (A)
                src_regions = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj1_pos_a],
                    shape_size=shape_size, padding=pad,
                )
                # Target: obj1 patches in corrupted image (B)
                tgt_regions = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj1_pos_b],
                    shape_size=shape_size, padding=pad,
                )
                src_indices = src_regions[0]
                tgt_indices = tgt_regions[0]

                # Cross-position: must have same number of patches
                if len(src_indices) == len(tgt_indices):
                    cross_groups.append((f"obj1_cross{suffix}", src_indices, tgt_indices))
                else:
                    print(f"  Warning: obj1_cross{suffix} patch count mismatch "
                          f"(src={len(src_indices)}, tgt={len(tgt_indices)}), skipping")

            # obj2 same-position with padding levels (center, identical in both images)
            if has_obj2_positions:
                obj2_pos_a = tuple(sample_a["obj2_position"])
                for pad, suffix in [(0, ""), (1, "_pad1"), (2, "_pad2")]:
                    obj2_regions = compute_image_patch_regions(
                        image_range=ranges["image"], grid_hw=grid_hw,
                        image_size=image_size, positions=[obj2_pos_a],
                        shape_size=shape_size, padding=pad,
                    )
                    cross_groups.append((f"obj2_same{suffix}", obj2_regions[0], obj2_regions[0]))

            for group_name, src_indices, tgt_indices in cross_groups:
                if group_name not in all_scores:
                    group_names.append(group_name)
                    all_scores[group_name] = {l: [] for l in layers_to_trace}
                    all_argmax[group_name] = {l: [] for l in layers_to_trace}

                for layer_idx in layers_to_trace:
                    p_patched, patched_argmax = run_with_patch(
                        model,
                        inputs_b,
                        clean_states,
                        patch_layer=layer_idx,
                        patch_token_indices=tgt_indices,
                        gt_token_id=gt_token_id,
                        source_token_indices=src_indices,
                    )

                    gap = p_clean - p_corrupted
                    if abs(gap) > 1e-6:
                        score = (p_patched - p_corrupted) / gap
                    else:
                        score = 0.0

                    all_scores[group_name][layer_idx].append(score)
                    all_argmax[group_name][layer_idx].append(patched_argmax)

        # 7. Both objects merged patch groups: obj1_cross + obj2_same at each padding level
        if both_patch_groups and has_obj1_positions and has_obj2_positions and ranges["image"] is not None:
            obj1_pos_a = tuple(sample_a["obj1_position"])
            obj1_pos_b = tuple(sample_b["obj1_position"])
            obj2_pos_a = tuple(sample_a["obj2_position"])

            n_img_tokens = ranges["image"][1] - ranges["image"][0]
            grid_hw = _get_image_grid(inputs_a, n_img_tokens)

            both_groups = []
            for pad, suffix in [(0, ""), (1, "_pad1"), (2, "_pad2")]:
                # obj1: cross-position (clean A position → corrupted B position)
                obj1_src = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj1_pos_a],
                    shape_size=shape_size, padding=pad,
                )[0]
                obj1_tgt = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj1_pos_b],
                    shape_size=shape_size, padding=pad,
                )[0]
                # obj2: same-position (center, identical in both)
                obj2_indices = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj2_pos_a],
                    shape_size=shape_size, padding=pad,
                )[0]

                if len(obj1_src) == len(obj1_tgt):
                    # Merge: src = obj1_src + obj2, tgt = obj1_tgt + obj2
                    merged_src = obj1_src + obj2_indices
                    merged_tgt = obj1_tgt + obj2_indices
                    both_groups.append((f"both_obj{suffix}", merged_src, merged_tgt))

            for group_name, src_indices, tgt_indices in both_groups:
                if group_name not in all_scores:
                    group_names.append(group_name)
                    all_scores[group_name] = {l: [] for l in layers_to_trace}
                    all_argmax[group_name] = {l: [] for l in layers_to_trace}

                for layer_idx in layers_to_trace:
                    p_patched, patched_argmax = run_with_patch(
                        model,
                        inputs_b,
                        clean_states,
                        patch_layer=layer_idx,
                        patch_token_indices=tgt_indices,
                        gt_token_id=gt_token_id,
                        source_token_indices=src_indices,
                    )

                    gap = p_clean - p_corrupted
                    if abs(gap) > 1e-6:
                        score = (p_patched - p_corrupted) / gap
                    else:
                        score = 0.0

                    all_scores[group_name][layer_idx].append(score)
                    all_argmax[group_name][layer_idx].append(patched_argmax)

        # 8. Random patch groups: obj1 core + N random non-object image tokens
        if random_patch_groups and has_obj1_positions and ranges["image"] is not None:
            obj1_pos_a = tuple(sample_a["obj1_position"])
            obj1_pos_b = tuple(sample_b["obj1_position"])

            n_img_tokens = ranges["image"][1] - ranges["image"][0]
            grid_hw = _get_image_grid(inputs_a, n_img_tokens)
            img_start = ranges["image"][0]

            # Core obj1 patches (no padding)
            obj1_src_core = compute_image_patch_regions(
                image_range=ranges["image"], grid_hw=grid_hw,
                image_size=image_size, positions=[obj1_pos_a],
                shape_size=shape_size, padding=0,
            )[0]
            obj1_tgt_core = compute_image_patch_regions(
                image_range=ranges["image"], grid_hw=grid_hw,
                image_size=image_size, positions=[obj1_pos_b],
                shape_size=shape_size, padding=0,
            )[0]

            # Obj1 padded patches (to know how many random tokens to add)
            obj1_src_pad1 = compute_image_patch_regions(
                image_range=ranges["image"], grid_hw=grid_hw,
                image_size=image_size, positions=[obj1_pos_a],
                shape_size=shape_size, padding=1,
            )[0]
            obj1_src_pad2 = compute_image_patch_regions(
                image_range=ranges["image"], grid_hw=grid_hw,
                image_size=image_size, positions=[obj1_pos_a],
                shape_size=shape_size, padding=2,
            )[0]

            # Exclude obj2 patches from random pool
            exclude_set = set(obj1_src_core)
            if has_obj2_positions:
                obj2_pos_a = tuple(sample_a["obj2_position"])
                obj2_indices = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj2_pos_a],
                    shape_size=shape_size, padding=0,
                )[0]
                exclude_set.update(obj2_indices)

            # Also exclude from target side
            exclude_tgt = set(obj1_tgt_core)
            if has_obj2_positions:
                exclude_tgt.update(obj2_indices)

            # Available random tokens (all image tokens except obj1 and obj2)
            all_img_indices = list(range(img_start, img_start + n_img_tokens))
            pool_src = [i for i in all_img_indices if i not in exclude_set]
            pool_tgt = [i for i in all_img_indices if i not in exclude_tgt]

            rng = np.random.RandomState(pair_id)  # reproducible per pair

            random_groups = []
            # obj1 + random padding
            for padded, suffix in [(obj1_src_pad1, "_rpad1"), (obj1_src_pad2, "_rpad2")]:
                n_extra = len(padded) - len(obj1_src_core)
                if n_extra <= 0 or n_extra > len(pool_src) or n_extra > len(pool_tgt):
                    continue
                rand_src = sorted(rng.choice(pool_src, size=n_extra, replace=False).tolist())
                rand_tgt = sorted(rng.choice(pool_tgt, size=n_extra, replace=False).tolist())
                merged_src = obj1_src_core + rand_src
                merged_tgt = obj1_tgt_core + rand_tgt
                random_groups.append((f"obj1_cross{suffix}", merged_src, merged_tgt))

            # obj2 + random padding (same-position, exclude obj1 from pool)
            if has_obj2_positions:
                obj2_core = obj2_indices  # already computed above (pad=0)
                obj2_pad1 = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj2_pos_a],
                    shape_size=shape_size, padding=1,
                )[0]
                obj2_pad2 = compute_image_patch_regions(
                    image_range=ranges["image"], grid_hw=grid_hw,
                    image_size=image_size, positions=[obj2_pos_a],
                    shape_size=shape_size, padding=2,
                )[0]
                # Pool for obj2: exclude both objects
                pool_obj2 = [i for i in all_img_indices
                             if i not in set(obj2_core) | set(obj1_src_core)]
                for padded, suffix in [(obj2_pad1, "_rpad1"), (obj2_pad2, "_rpad2")]:
                    n_extra = len(padded) - len(obj2_core)
                    if n_extra <= 0 or n_extra > len(pool_obj2):
                        continue
                    rand_idx = sorted(rng.choice(pool_obj2, size=n_extra, replace=False).tolist())
                    merged = obj2_core + rand_idx
                    random_groups.append((f"obj2_same{suffix}", merged, merged))

                # both_obj + random padding
                if len(obj1_src_core) == len(obj1_tgt_core):
                    both_core_src = obj1_src_core + obj2_core
                    both_core_tgt = obj1_tgt_core + obj2_core
                    both_pad1_src = obj1_src_pad1 + obj2_pad1
                    both_pad2_src = compute_image_patch_regions(
                        image_range=ranges["image"], grid_hw=grid_hw,
                        image_size=image_size, positions=[obj1_pos_a],
                        shape_size=shape_size, padding=2,
                    )[0] + obj2_pad2
                    # Pool: exclude both objects
                    pool_both_src = [i for i in all_img_indices
                                     if i not in set(obj1_src_core) | set(obj2_core)]
                    pool_both_tgt = [i for i in all_img_indices
                                     if i not in set(obj1_tgt_core) | set(obj2_core)]
                    for padded_src, suffix in [(both_pad1_src, "_rpad1"), (both_pad2_src, "_rpad2")]:
                        n_extra = len(padded_src) - len(both_core_src)
                        if n_extra <= 0 or n_extra > len(pool_both_src) or n_extra > len(pool_both_tgt):
                            continue
                        rand_src = sorted(rng.choice(pool_both_src, size=n_extra, replace=False).tolist())
                        rand_tgt = sorted(rng.choice(pool_both_tgt, size=n_extra, replace=False).tolist())
                        merged_src = both_core_src + rand_src
                        merged_tgt = both_core_tgt + rand_tgt
                        random_groups.append((f"both_obj{suffix}", merged_src, merged_tgt))

            for group_name, src_indices, tgt_indices in random_groups:
                if group_name not in all_scores:
                    group_names.append(group_name)
                    all_scores[group_name] = {l: [] for l in layers_to_trace}
                    all_argmax[group_name] = {l: [] for l in layers_to_trace}

                for layer_idx in layers_to_trace:
                    p_patched, patched_argmax = run_with_patch(
                        model,
                        inputs_b,
                        clean_states,
                        patch_layer=layer_idx,
                        patch_token_indices=tgt_indices,
                        gt_token_id=gt_token_id,
                        source_token_indices=src_indices,
                    )

                    gap = p_clean - p_corrupted
                    if abs(gap) > 1e-6:
                        score = (p_patched - p_corrupted) / gap
                    else:
                        score = 0.0

                    all_scores[group_name][layer_idx].append(score)
                    all_argmax[group_name][layer_idx].append(patched_argmax)

        # Free memory
        del clean_states
        torch.cuda.empty_cache()

        # Print progress every 10 pairs
        if (pair_id + 1) % 10 == 0:
            print(
                f"  Pair {pair_id + 1}/{n_pairs} | P(clean)={p_clean:.3f} P(corrupted)={p_corrupted:.3f}"
            )

    n_evaluated = n_pairs - n_skipped
    if n_skipped > 0:
        # Count skip reasons from metadata
        skip_reasons = {}
        for m in pair_metadata:
            reason = m.get("skipped")
            if reason is not None:
                key = reason.split("_")[0] if reason.startswith("tiny_gap") else reason
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
        reason_str = ", ".join(f"{k}={v}" for k, v in sorted(skip_reasons.items()))
        print(f"\nSkipped {n_skipped}/{n_pairs} pairs ({reason_str}). "
              f"Evaluating {n_evaluated} pairs.")

    # Average scores across pairs
    heatmap = {}
    heatmap_std = {}
    # Argmax classification: for each (group, layer), fraction of pairs where
    # patched model predicts clean answer / corrupted answer / other
    argmax_fractions = {}

    for group_name in group_names:
        heatmap[group_name] = {}
        heatmap_std[group_name] = {}
        argmax_fractions[group_name] = {}
        for layer_idx in layers_to_trace:
            scores = all_scores[group_name][layer_idx]
            heatmap[group_name][layer_idx] = float(np.mean(scores)) if scores else 0.0
            heatmap_std[group_name][layer_idx] = float(np.std(scores)) if scores else 0.0

            # Classify argmax predictions
            argmax_ids = all_argmax[group_name][layer_idx]
            if argmax_ids:
                n = len(argmax_ids)
                n_clean = sum(1 for a, gt in zip(argmax_ids, pair_gt_ids) if a == gt)
                # Only count as "corrupted" if it's a different token from clean
                n_corrupted = sum(1 for a, gt, corr in zip(argmax_ids, pair_gt_ids, pair_corrupted_ids)
                                  if a == corr and a != gt)
                n_other = n - n_clean - n_corrupted
                argmax_fractions[group_name][layer_idx] = {
                    "clean": round(n_clean / n, 4),
                    "corrupted": round(n_corrupted / n, 4),
                    "other": round(n_other / n, 4),
                }
            else:
                argmax_fractions[group_name][layer_idx] = {
                    "clean": 0.0, "corrupted": 0.0, "other": 0.0,
                }

    # Per-option heatmaps (logit-based restoration scores)
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
        "layer_step": layer_step,
        "n_layers": n_layers,
        "dataset_name": dataset_name,
        "question_format": question_format,
    }

    if option_heatmaps:
        results["option_heatmaps"] = option_heatmaps

    return results


# =============================================================================
# Unpaired causal tracing — corruption via synthetic images
# =============================================================================

def generate_corrupted_image(corruption_type, image_size, reference_images=None,
                             image_path=None, seed=42):
    """Generate a corrupted image for unpaired causal tracing.

    Args:
        corruption_type: One of 'white', 'black', 'noise', 'average', 'random', 'file'
        image_size: (width, height) tuple
        reference_images: List of PIL images (for 'average' and 'random' modes)
        image_path: Path to a specific image file (for 'file' mode)
        seed: Random seed for reproducibility

    Returns:
        PIL Image
    """
    w, h = image_size
    rng = np.random.RandomState(seed)

    if corruption_type == "file":
        if not image_path:
            raise ValueError("'file' corruption requires image_path")
        img = Image.open(image_path).convert("RGB")
        return img.resize((w, h), Image.BILINEAR)
    elif corruption_type == "white":
        return Image.new("RGB", (w, h), (255, 255, 255))
    elif corruption_type == "black":
        return Image.new("RGB", (w, h), (0, 0, 0))
    elif corruption_type == "noise":
        arr = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    elif corruption_type == "random":
        if not reference_images:
            raise ValueError("'random' corruption requires reference_images")
        idx = rng.randint(0, len(reference_images))
        img = reference_images[idx]
        return img.resize((w, h), Image.BILINEAR)
    elif corruption_type == "average":
        if not reference_images:
            raise ValueError("'average' corruption requires reference_images")
        n = min(50, len(reference_images))
        indices = rng.choice(len(reference_images), size=n, replace=False)
        acc = np.zeros((h, w, 3), dtype=np.float64)
        for i in indices:
            img = reference_images[i].resize((w, h), Image.BILINEAR)
            acc += np.array(img, dtype=np.float64)
        acc /= n
        return Image.fromarray(acc.astype(np.uint8))
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")


def run_causal_tracing_unpaired(
    dataset_path,
    dataset_name,
    model_path,
    n_samples=50,
    layer_step=1,
    group_specs=None,
    combos=False,
    question_format="forced_choice",
    task_type="spatial_relative",
    perspective=None,
    corruption_type="white",
    corruption_source_path=None,
    corruption_source_name=None,
    corruption_image_path=None,
    start_from=0,
):
    """Run causal tracing with synthetic image corruption (no pairs needed).

    Instead of using a position-swapped paired image as corruption, uses a
    synthetic image (white, black, noise, average, or random). This allows
    causal tracing on ANY dataset, not just paired ones.

    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of dataset
        model_path: Path to model
        n_samples: Number of samples to process
        layer_step: Process every N-th layer
        group_specs: Specific groups to compute (None = defaults)
        combos: If True, add pairwise + leave-one-out groups
        question_format: Question format
        task_type: Task type
        perspective: POV perspective
        corruption_type: 'white', 'black', 'noise', 'random', or 'average'
        corruption_source_path: Dataset path for 'random'/'average' corruption images
        corruption_source_name: Dataset name for 'random'/'average' corruption images

    Returns:
        Results dict with heatmap data (same format as paired version)
    """
    model, processor = load_model(model_path)
    dataset, prepositions = load_dataset(dataset_path, dataset_name,
                                         task_type=task_type, perspective=perspective)

    n_layers = len(get_language_layers(model))
    layers_to_trace = list(range(0, n_layers, layer_step))
    print(f"Tracing {len(layers_to_trace)} layers (step={layer_step}): "
          f"{layers_to_trace[0]}-{layers_to_trace[-1]}")

    n_samples = min(n_samples, len(dataset))
    print(f"Processing {n_samples} samples (corruption={corruption_type})")

    # Load reference images for 'random' or 'average' corruption
    reference_images = None
    if corruption_type in ("random", "average"):
        src_path = corruption_source_path or dataset_path
        src_name = corruption_source_name or dataset_name
        print(f"Loading reference images from {src_name} for {corruption_type} corruption...")
        ref_dataset, _ = load_dataset(src_path, src_name, task_type=task_type)
        reference_images = [ref_dataset[i]["image"] for i in range(len(ref_dataset))]
        print(f"  Loaded {len(reference_images)} reference images")

    # Generate corrupted image per sample to match dimensions (different images
    # produce different vision token counts, so sequence lengths must match).
    # Cache by (width, height) to avoid regenerating for same-sized images.
    _corrupted_cache = {}

    def get_corrupted_image(w, h):
        key = (w, h)
        if key not in _corrupted_cache:
            _corrupted_cache[key] = generate_corrupted_image(
                corruption_type, (w, h), reference_images=reference_images,
                image_path=corruption_image_path, seed=42
            )
        return _corrupted_cache[key]

    print(f"Corrupted image: {corruption_type} (per-sample, matching clean image size)")

    # Accumulators (same structure as paired version)
    all_scores = {}
    all_argmax = {}
    pair_gt_ids = []        # gt_token_id per sample
    pair_corrupted_ids = [] # corrupted argmax per sample
    pair_metadata = []
    group_names = None
    n_skipped = 0

    end_idx = start_from + n_samples
    if start_from > 0:
        print(f"Starting from sample {start_from} (processing {start_from}-{end_idx - 1})")

    for idx in tqdm(range(start_from, end_idx), desc=f"Causal tracing ({corruption_type})"):
        sample = dataset[idx]
        image = sample["image"]
        objects = sample["objects"]
        gt = sample["preposition"]

        question = create_question(objects, prepositions,
                                   question_format=question_format,
                                   task_type=task_type, perspective=perspective)

        gt_token_ids = processor.tokenizer.encode(gt, add_special_tokens=False)
        gt_token_id = gt_token_ids[0]

        # Generate corrupted image matching this sample's dimensions
        corrupted_image = get_corrupted_image(image.width, image.height)

        # Prepare clean and corrupted inputs (same question, different image)
        inputs_clean = prepare_inputs(processor, image, question).to(model.device)
        inputs_corrupt = prepare_inputs(processor, corrupted_image, question).to(model.device)

        # Token ranges (from clean input — same tokenization for both)
        ranges = find_token_ranges(inputs_clean.input_ids, processor.tokenizer)
        cluster_ranges = find_cluster_ranges(
            inputs_clean.input_ids, processor.tokenizer, objects,
            ground_truth=gt, prepositions=prepositions,
        )

        # Define token groups
        token_groups = define_token_groups(
            ranges, cluster_ranges, combos=combos, group_specs=group_specs,
        )

        # Resolve last token index
        seq_len = inputs_clean.input_ids.shape[1]
        token_groups = [
            (name, [i if i >= 0 else seq_len + i for i in indices])
            for name, indices in token_groups
        ]

        if group_names is None:
            group_names = [g[0] for g in token_groups]
            for name in group_names:
                all_scores[name] = {l: [] for l in layers_to_trace}
                all_argmax[name] = {l: [] for l in layers_to_trace}

        # 1. Clean run
        layers_to_collect = sorted(set(layers_to_trace) | {0})
        clean_states, clean_logits = collect_hidden_states(
            model, inputs_clean, layers_to_collect
        )
        clean_probs = torch.softmax(clean_logits, dim=-1)
        p_clean = clean_probs[gt_token_id].item()

        # Skip samples where model gets clean run wrong
        clean_argmax = clean_logits.argmax().item()
        if clean_argmax != gt_token_id:
            clean_answer = processor.tokenizer.decode([clean_argmax]).strip().lower()
            gt_answer = processor.tokenizer.decode([gt_token_id]).strip().lower()
            if clean_answer == gt_answer:
                gt_token_id = clean_argmax
                p_clean = clean_probs[gt_token_id].item()
            else:
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  Skipping sample {idx}: model answered '{clean_answer}' "
                          f"instead of '{gt_answer}' on clean run")
                continue

        # 2. Corrupted run
        p_corrupted, corrupted_argmax = get_answer_prob(model, inputs_corrupt, gt_token_id)
        pair_gt_ids.append(gt_token_id)
        pair_corrupted_ids.append(corrupted_argmax)

        # Save per-sample metadata (including P for all answer options)
        corrupted_logits = _get_logits(model, inputs_corrupt)
        corrupted_probs = torch.softmax(corrupted_logits, dim=-1)
        p_all_answers = {}
        for prep in prepositions:
            prep_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
            p_all_answers[prep] = round(corrupted_probs[prep_ids[0]].item(), 6)

        pair_metadata.append({
            "sample_idx": idx,
            "gt_token_id": gt_token_id,
            "gt_token": processor.tokenizer.decode([gt_token_id]).strip(),
            "gt_label": gt,
            "corrupted_token_id": corrupted_argmax,
            "corrupted_token": processor.tokenizer.decode([corrupted_argmax]).strip(),
            "p_clean": round(p_clean, 6),
            "p_corrupted": round(p_corrupted, 6),
            "p_corrupted_all": p_all_answers,
        })

        # 3. Patched runs
        for group_name, token_indices in token_groups:
            for layer_idx in layers_to_trace:
                p_patched, patched_argmax = run_with_patch(
                    model, inputs_corrupt, clean_states,
                    patch_layer=layer_idx,
                    patch_token_indices=token_indices,
                    gt_token_id=gt_token_id,
                )

                gap = p_clean - p_corrupted
                if abs(gap) > 1e-6:
                    score = (p_patched - p_corrupted) / gap
                else:
                    score = 0.0

                all_scores[group_name][layer_idx].append(score)
                all_argmax[group_name][layer_idx].append(patched_argmax)

        # 4. Cross-layer patching: image(from_L0)
        img_indices = None
        if group_specs is None:
            for name, indices in token_groups:
                if name == "image":
                    img_indices = indices
                    break

        if img_indices is not None:
            cross_name = "image(from_L0)"
            if cross_name not in all_scores:
                group_names.append(cross_name)
                all_scores[cross_name] = {l: [] for l in layers_to_trace}
                all_argmax[cross_name] = {l: [] for l in layers_to_trace}

            for layer_idx in layers_to_trace:
                p_patched, patched_argmax = run_with_patch(
                    model, inputs_corrupt, clean_states,
                    patch_layer=layer_idx,
                    patch_token_indices=img_indices,
                    gt_token_id=gt_token_id,
                    source_layer=0,
                )
                gap = p_clean - p_corrupted
                score = (p_patched - p_corrupted) / gap if abs(gap) > 1e-6 else 0.0
                all_scores[cross_name][layer_idx].append(score)
                all_argmax[cross_name][layer_idx].append(patched_argmax)

        del clean_states
        torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            print(f"  Sample {idx + 1}/{n_samples} | P(clean)={p_clean:.3f} "
                  f"P(corrupted)={p_corrupted:.3f}")

    n_evaluated = n_samples - n_skipped
    if n_skipped > 0:
        print(f"\nSkipped {n_skipped}/{n_samples} samples (model incorrect on clean run). "
              f"Evaluating {n_evaluated} samples.")

    # Aggregate (same code as paired version)
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
                # Only count as "corrupted" if it's a different token from clean
                n_corrupted = sum(1 for a, gt, corr in zip(argmax_ids, pair_gt_ids, pair_corrupted_ids)
                                  if a == corr and a != gt)
                n_other = n - n_clean - n_corrupted
                argmax_fractions[group_name][layer_idx] = {
                    "clean": round(n_clean / n, 4),
                    "corrupted": round(n_corrupted / n, 4),
                    "other": round(n_other / n, 4),
                }
            else:
                argmax_fractions[group_name][layer_idx] = {
                    "clean": 0.0, "corrupted": 0.0, "other": 0.0,
                }

    results = {
        "heatmap": heatmap,
        "heatmap_std": heatmap_std,
        "argmax_fractions": argmax_fractions,
        "pair_metadata": pair_metadata,
        "group_names": group_names,
        "layers": layers_to_trace,
        "n_pairs": n_samples,  # kept as n_pairs for compatibility
        "n_skipped": n_skipped,
        "n_evaluated": n_evaluated,
        "layer_step": layer_step,
        "n_layers": n_layers,
        "dataset_name": dataset_name,
        "question_format": question_format,
        "corruption_type": corruption_type,
    }

    return results


def print_heatmap(results):
    """Print heatmap as formatted table."""
    heatmap = results["heatmap"]
    layers = results["layers"]
    group_names = results["group_names"]

    name_w = max(20, max(len(n) for n in group_names) + 2)
    col_w = 7

    header = f"{'token group':<{name_w}}"
    for l in layers:
        header += f" {l:>{col_w}}"
    print(f"\n{'=' * len(header)}")
    print("  Causal Tracing: Restoration Scores")
    print(f"  (1.0 = fully recovers clean answer, 0.0 = no recovery)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for name in group_names:
        row = f"{name:<{name_w}}"
        for l in layers:
            val = heatmap[name].get(l, 0.0)
            row += f" {val:>{col_w}.3f}"
        print(row)

    print("-" * len(header))


def plot_heatmap(results, output_path):
    """Plot causal tracing heatmap."""
    heatmap = results["heatmap"]
    layers = results["layers"]
    group_names = results["group_names"]

    # Build matrix
    matrix = np.zeros((len(group_names), len(layers)))
    for i, name in enumerate(group_names):
        for j, l in enumerate(layers):
            matrix[i, j] = heatmap[name].get(l, 0.0)

    fig, ax = plt.subplots(
        figsize=(max(12, len(layers) * 0.5), max(6, len(group_names) * 0.5))
    )

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Token Group", fontsize=11)
    ax.set_title(
        f"Causal Tracing: Restoration Scores ({results['dataset_name']}, {results['n_pairs']} pairs)",
        fontsize=12,
        fontweight="bold",
    )

    # Add value annotations
    for i in range(len(group_names)):
        for j in range(len(layers)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color
            )

    plt.colorbar(im, ax=ax, label="Restoration Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def merge_results(existing, new_results):
    """Merge new causal tracing results into an existing results dict.

    Adds new groups to the heatmap and group_names. Existing groups are
    preserved (not overwritten). Updates n_pairs to the new value if different.

    Args:
        existing: Existing results dict (loaded from JSON).
        new_results: New results dict from run_causal_tracing.

    Returns:
        Merged results dict.
    """
    merged = existing.copy()
    merged["heatmap"] = dict(existing["heatmap"])
    merged["heatmap_std"] = dict(existing.get("heatmap_std", {}))
    merged["argmax_fractions"] = dict(existing.get("argmax_fractions", {}))

    existing_names = set(existing.get("group_names", []))
    added = []

    for name in new_results["group_names"]:
        if name not in existing_names:
            merged["heatmap"][name] = new_results["heatmap"][name]
            if "heatmap_std" in new_results and name in new_results["heatmap_std"]:
                merged["heatmap_std"][name] = new_results["heatmap_std"][name]
            if "argmax_fractions" in new_results and name in new_results["argmax_fractions"]:
                merged["argmax_fractions"][name] = new_results["argmax_fractions"][name]
            added.append(name)

    # Append new names to group_names (preserve original order)
    merged["group_names"] = existing.get("group_names", []) + added

    if added:
        print(f"Merged {len(added)} new groups: {added}")
    else:
        print("No new groups to merge (all already exist).")

    return merged


def merge_samples(existing, new_results):
    """Merge additional samples into existing results (weighted average).

    Unlike merge_results (which adds new groups), this adds more samples to
    the same groups, combining heatmap scores via weighted average.

    Args:
        existing: Existing results dict (loaded from JSON).
        new_results: New results dict from a continuation run.

    Returns:
        Merged results dict with updated n_evaluated, heatmap, pair_metadata.
    """
    old_n = existing["n_evaluated"]
    new_n = new_results["n_evaluated"]
    total_n = old_n + new_n

    merged = existing.copy()
    merged["heatmap"] = {}
    merged["heatmap_std"] = {}
    merged["argmax_fractions"] = {}

    for group_name in existing["group_names"]:
        merged["heatmap"][group_name] = {}
        merged["heatmap_std"][group_name] = {}
        merged["argmax_fractions"][group_name] = {}

        for layer in existing["layers"]:
            lk = str(layer)
            old_mean = existing["heatmap"][group_name].get(lk, existing["heatmap"][group_name].get(layer, 0.0))
            new_mean = new_results["heatmap"][group_name].get(lk, new_results["heatmap"][group_name].get(layer, 0.0))
            combined_mean = (old_mean * old_n + new_mean * new_n) / total_n
            merged["heatmap"][group_name][layer] = round(combined_mean, 6)

            # Approximate combined std (using pooled variance)
            old_std = existing.get("heatmap_std", {}).get(group_name, {}).get(lk,
                      existing.get("heatmap_std", {}).get(group_name, {}).get(layer, 0.0))
            new_std = new_results.get("heatmap_std", {}).get(group_name, {}).get(lk,
                      new_results.get("heatmap_std", {}).get(group_name, {}).get(layer, 0.0))
            if total_n > 0:
                pooled_var = ((old_n - 1) * old_std**2 + (new_n - 1) * new_std**2 +
                              old_n * new_n / total_n * (old_mean - new_mean)**2) / (total_n - 1) if total_n > 1 else 0.0
                merged["heatmap_std"][group_name][layer] = round(pooled_var**0.5, 6)

            # Merge argmax fractions (weighted average)
            old_af = existing.get("argmax_fractions", {}).get(group_name, {}).get(lk,
                     existing.get("argmax_fractions", {}).get(group_name, {}).get(layer, {}))
            new_af = new_results.get("argmax_fractions", {}).get(group_name, {}).get(lk,
                     new_results.get("argmax_fractions", {}).get(group_name, {}).get(layer, {}))
            if old_af and new_af:
                merged["argmax_fractions"][group_name][layer] = {
                    k: round((old_af.get(k, 0) * old_n + new_af.get(k, 0) * new_n) / total_n, 4)
                    for k in set(list(old_af.keys()) + list(new_af.keys()))
                }

    # Concatenate pair_metadata
    merged["pair_metadata"] = existing.get("pair_metadata", []) + new_results.get("pair_metadata", [])

    # Update counts
    merged["n_pairs"] = existing.get("n_pairs", old_n) + new_results.get("n_pairs", new_n)
    merged["n_skipped"] = existing.get("n_skipped", 0) + new_results.get("n_skipped", 0)
    merged["n_evaluated"] = total_n

    print(f"Merged samples: {old_n} + {new_n} = {total_n} evaluated")
    return merged


def save_results(results, output_dir=None, output_file=None):
    """Save results to JSON and plot.

    Args:
        results: Results dict.
        output_dir: Directory to save into (creates causal_tracing_results.json).
        output_file: Explicit file path to save to (overrides output_dir).
    """
    if output_file is not None:
        json_path = Path(output_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "causal_tracing_results.json"
    else:
        raise ValueError("Either output_dir or output_file must be provided")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Save heatmap plot next to the JSON
    plot_path = json_path.with_suffix(".png")
    plot_heatmap(results, plot_path)


def main():
    parser = argparse.ArgumentParser(
        description="Causal tracing experiment for spatial reasoning in VLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--dataset-path", type=str, help="Base path to datasets")
    parser.add_argument("--dataset-name", type=str, help="Name of paired dataset")
    parser.add_argument("--model-path", type=str, help="Path to Qwen3-VL model")
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=20,
        help="Number of pairs to process (default: 20)",
    )
    parser.add_argument(
        "--layer-step",
        type=int,
        default=1,
        help="Process every N-th layer (default: 1 = all 32 layers). Use 2 or 4 for faster runs.",
    )
    parser.add_argument(
        "--shape-size",
        type=int,
        default=None,
        help="Object diameter in pixels (for region-based patching). "
             "If not provided, reads from dataset metadata.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Image size in pixels (default: 448). Used for region-based patching.",
    )
    parser.add_argument(
        "--combos", action="store_true",
        help="Run combinatorial patching: all pairwise cluster combos (36) + "
             "leave-one-out from all_text (9). Tests which cluster subsets are "
             "sufficient/necessary for restoring spatial signal.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated group specs to compute. Supports merged clusters "
             "(e.g. 'object1+in_relation+object2') and complements "
             "(e.g. 'all_text-object1-in_relation-object2'). "
             "When used with --append-to, only these groups are computed and added.",
    )
    parser.add_argument(
        "--append-to",
        type=str,
        default=None,
        help="Path to existing causal_tracing_results.json. New groups from "
             "--groups will be merged into this file. Groups that already exist "
             "are skipped.",
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
        "--question-override", type=str, default=None,
        help="Override the question text. Format suffix (Answer with...) is added automatically.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="spatial_relative",
        choices=["spatial_relative", "spatial_absolute", "recognition", "attribute_chain", "attribute_shape", "spatial_or"],
        help="Task type: spatial_relative (2 objects, default), "
             "spatial_absolute (1 object, absolute position), "
             "recognition (what object is in image), "
             "attribute_chain (color of object at direction from reference), "
             "attribute_shape (shape of object at direction from reference).",
    )
    parser.add_argument(
        "--perspective",
        type=str,
        default=None,
        choices=["camera", "addressee"],
        help="Perspective prefix for POV questions (COMFORT dataset). "
             "camera: 'From the camera's viewpoint, ...', "
             "addressee: 'From the woman's viewpoint, ...'.",
    )
    parser.add_argument(
        "--patch-groups", action="store_true",
        help="Add cross-position patch groups: obj1_cross (exact), obj1_cross_pad1, "
             "obj1_cross_pad2, obj2_same. Requires dataset with position data. "
             "Uses clean obj1 representation at corrupted obj1 position.",
    )
    parser.add_argument(
        "--both-patch-groups", action="store_true",
        help="Add merged obj1+obj2 patch groups: both_obj (exact), both_obj_pad1, "
             "both_obj_pad2. Tests whether the rest of the image carries other info.",
    )
    parser.add_argument(
        "--random-patch-groups", action="store_true",
        help="Add obj1 + random non-object image tokens: obj1_cross_rpad1, "
             "obj1_cross_rpad2. Same token count as pad1/pad2 but random positions "
             "(excluding obj2). Tests if specific surrounding patches matter.",
    )
    # Unpaired corruption modes
    parser.add_argument(
        "--corruption", type=str, default=None,
        choices=["white", "black", "noise", "random", "average", "file"],
        help="Corruption mode for unpaired causal tracing (no pair_id needed). "
             "white/black: solid color. noise: gaussian. "
             "random: one fixed unrelated image from --corruption-source. "
             "average: mean of 50 images from --corruption-source. "
             "file: use a specific image from --corruption-image. "
             "If not set, uses standard paired mode (requires pair_id).",
    )
    parser.add_argument(
        "--corruption-source", type=str, default=None,
        help="Dataset name for 'random'/'average' corruption images "
             "(default: same dataset).",
    )
    parser.add_argument(
        "--corruption-image", type=str, default=None,
        help="Path to a specific image file for 'file' corruption mode.",
    )

    parser.add_argument(
        "--continue-from", type=str, default=None, metavar="FILE",
        help="Path to existing causal_tracing_results.json. Adds more samples to "
             "the existing results using weighted-average merging. Use with "
             "--n-pairs to set the NEW sample count (will start from sample "
             "N_existing in the dataset). Heatmap scores are merged via weighted "
             "average; pair_metadata lists are concatenated.",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--save-option-logits", action="store_true",
                        help="Save per-option restoration scores (logits for all valid answers at each layer)")
    parser.add_argument(
        "--results-subdir", type=str, default=None,
        help="Override results subdirectory (e.g., 'causal_tracing_llava05b').",
    )

    args = parser.parse_args()

    # Parse group specs
    group_specs = None
    if args.groups:
        group_specs = [g.strip() for g in args.groups.split(",")]
    elif (args.patch_groups or args.both_patch_groups or args.random_patch_groups) and args.append_to:
        # Only patch groups, no standard text/image groups
        group_specs = []

    # Load existing results if appending
    existing_results = None
    if args.append_to:
        append_path = Path(args.append_to)
        if not append_path.exists():
            parser.error(f"--append-to file not found: {append_path}")
        with open(append_path) as f:
            existing_results = json.load(f)
        print(f"Loaded existing results from {append_path} "
              f"({len(existing_results.get('group_names', []))} groups)")

        # Filter out groups that already exist
        if group_specs:
            existing_names = set(existing_results.get("group_names", []))
            new_specs = [g for g in group_specs if g not in existing_names]
            skipped = [g for g in group_specs if g in existing_names]
            if skipped:
                print(f"Skipping already-existing groups: {skipped}")
            if not new_specs:
                print("All requested groups already exist. Nothing to compute.")
                print_heatmap(existing_results)
                return
            group_specs = new_specs
        elif not (args.combos or args.patch_groups or args.both_patch_groups or args.random_patch_groups):
            parser.error("--append-to requires --groups, --combos, --patch-groups, --both-patch-groups, or --random-patch-groups.")

    # Load --continue-from results
    continue_results = None
    start_from = 0
    if args.continue_from:
        continue_path = Path(args.continue_from)
        if not continue_path.exists():
            parser.error(f"--continue-from file not found: {continue_path}")
        with open(continue_path) as f:
            continue_results = json.load(f)
        old_n = continue_results["n_evaluated"]
        old_skipped = continue_results.get("n_skipped", 0)
        start_from = old_n + old_skipped
        print(f"Continuing from existing results: {old_n} evaluated, "
              f"{old_skipped} skipped → starting from sample {start_from}")

    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)

    if args.dataset_path is not None:
        config["dataset_path"] = args.dataset_path
    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    if args.model_path is not None:
        config["model_path"] = args.model_path

    required = ["dataset_path", "dataset_name", "model_path"]
    missing = [k for k in required if k not in config]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    # Run
    if args.corruption is not None:
        # Unpaired mode: synthetic corruption
        results = run_causal_tracing_unpaired(
            dataset_path=config["dataset_path"],
            dataset_name=config["dataset_name"],
            model_path=config["model_path"],
            n_samples=args.n_pairs,  # reuse --n-pairs as sample count
            layer_step=args.layer_step,
            group_specs=group_specs,
            combos=args.combos,
            question_format=args.question_format,
            task_type=args.task_type,
            perspective=args.perspective,
            corruption_type=args.corruption,
            corruption_source_path=config["dataset_path"],
            corruption_source_name=args.corruption_source,
            corruption_image_path=args.corruption_image,
            start_from=start_from,
        )
    else:
        # Paired mode: standard causal tracing
        results = run_causal_tracing(
            dataset_path=config["dataset_path"],
            dataset_name=config["dataset_name"],
            model_path=config["model_path"],
            n_pairs=args.n_pairs,
            layer_step=args.layer_step,
            shape_size=args.shape_size,
            image_size=args.image_size,
            combos=args.combos,
            group_specs=group_specs,
            question_format=args.question_format,
            patch_groups=args.patch_groups,
            both_patch_groups=args.both_patch_groups,
            random_patch_groups=args.random_patch_groups,
            task_type=args.task_type,
            perspective=args.perspective,
            save_option_logits=args.save_option_logits,
            question_override=args.question_override,
        )

    # Merge with existing samples if continuing
    if continue_results is not None:
        results = merge_samples(continue_results, results)

    # Merge with existing if appending groups
    if existing_results is not None:
        results = merge_results(existing_results, results)

    # Print
    print_heatmap(results)

    # Save
    if not args.no_save:
        if args.append_to:
            save_results(results, output_file=args.append_to)
        else:
            suffix = "_combos" if args.combos else ""
            fmt_suffix = f"_{args.question_format}" if args.question_format != "forced_choice" else ""
            task_suffix = f"_{args.task_type}" if args.task_type != "spatial_relative" else ""
            corruption_suffix = f"_{args.corruption}" if args.corruption else ""
            default_base = "corrupted_tracing" if args.corruption else "causal_tracing"
            results_base = args.results_subdir or default_base
            output_dir = (
                Path(__file__).parent.parent
                / "results"
                / results_base
                / f"{config['dataset_name']}{suffix}{fmt_suffix}{task_suffix}{corruption_suffix}"
            )
            save_results(results, output_dir=output_dir)


if __name__ == "__main__":
    main()
