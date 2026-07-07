import random

import torch


def get_language_layers(model):
    """
    Get the language model layers from a VLM, handling different architectures.

    Supports:
        - Qwen3-VL: model.model.language_model.layers
        - LLaVA-OneVision / LLaVA-1.5 / LLaVA-1.6: model.language_model.model.layers

    Returns:
        nn.ModuleList of transformer layers
    """
    # Qwen3-VL path: model.model is Qwen3VLModel, language_model stores layers directly
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
    # LLaVA family path: model.language_model is the LLM (LLaMA/Mistral/Qwen2)
    # layers are at model.language_model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        lm = model.language_model.model
        if hasattr(lm, "layers"):
            return lm.layers
    raise AttributeError(
        f"Cannot find language model layers in {type(model).__name__}. "
        "Expected model.model.language_model.layers or model.language_model.model.layers"
    )


def add_block_to_attention_mask(attn_mask, q_idx, k_idx):
    """
    Add blocking (-inf) to attention mask for specific query-key pairs.

    This is a pre-softmax intervention: adding -inf to the mask ensures
    those attention weights become 0 after softmax (no manual renormalization needed).

    Args:
        attn_mask: Attention mask tensor [B, 1, q_len, k_len]
                  (additive mask: 0 = keep, -inf = block)
        q_idx: Query indices to block (int or list of ints)
        k_idx: Key indices to block (slice object or list of ints)

    Returns:
        Modified attention mask with blocks added
    """
    B = attn_mask.shape[0]
    q_len = attn_mask.shape[-2]
    k_len = attn_mask.shape[-1]

    dtype = attn_mask.dtype
    device = attn_mask.device
    neginf = -1e9 if dtype in (torch.float16, torch.bfloat16) else -1e30

    # Create additive mask
    extra = torch.zeros((B, 1, q_len, k_len), device=device, dtype=dtype)

    # Convert q_idx to list if needed
    if isinstance(q_idx, int):
        q_idx = [q_idx]

    # Block specified query-key pairs
    for q in q_idx:
        extra[:, :, q, k_idx] = neginf

    return attn_mask + extra


def add_boost_to_attention_mask(attn_mask, q_idx, k_idx, alpha):
    """
    Add boost (+alpha) to attention mask for specific query-key pairs.

    This is a pre-softmax intervention: adding +alpha to the mask increases
    those attention weights after softmax (softmax auto-renormalizes).

    Args:
        attn_mask: Attention mask tensor [B, 1, q_len, k_len]
        q_idx: Query indices to boost (int or list of ints)
        k_idx: Key indices to boost (slice object or list of ints)
        alpha: Boost value (positive float)

    Returns:
        Modified attention mask with boosts added
    """
    B = attn_mask.shape[0]
    q_len = attn_mask.shape[-2]
    k_len = attn_mask.shape[-1]

    dtype = attn_mask.dtype
    device = attn_mask.device

    extra = torch.zeros((B, 1, q_len, k_len), device=device, dtype=dtype)

    if isinstance(q_idx, int):
        q_idx = [q_idx]

    for q in q_idx:
        extra[:, :, q, k_idx] = alpha

    return attn_mask + extra


def install_attention_boost(model, q_indices, k_indices, alpha, layer_range):
    """
    Install pre-hooks that boost attention scores at specific (query, key) positions.

    Generic amplification hook: adds +alpha to pre-softmax attention scores
    for the specified query→key pairs at the given layers.

    Args:
        model: VLM model
        q_indices: List of query token indices (or single int)
        k_indices: Key token indices (slice or list of ints)
        alpha: Boost value (positive float)
        layer_range: (start, end) tuple for half-open interval [start, end)

    Returns:
        Tuple of (hooks_list, stats_dict)
    """
    hooks = []
    stats = {"called": 0, "had_mask": 0, "alpha": alpha,
             "layer_range": list(layer_range)}

    if isinstance(q_indices, int):
        q_indices = [q_indices]

    def make_prehook(layer_idx):
        def prehook(module, args, kwargs):
            stats["called"] += 1
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is None:
                return
            stats["had_mask"] += 1

            q_len = attn_mask.shape[-2]

            # Resolve negative indices
            resolved_q = []
            for q in q_indices:
                resolved_q.append(q if q >= 0 else q_len + q)

            kwargs["attention_mask"] = add_boost_to_attention_mask(
                attn_mask, q_idx=resolved_q, k_idx=k_indices, alpha=alpha
            )

        return prehook

    layers = get_language_layers(model)

    for i, layer in enumerate(layers):
        if layer_range[0] <= i < layer_range[1]:
            handle = layer.self_attn.register_forward_pre_hook(
                make_prehook(i), with_kwargs=True
            )
            hooks.append(handle)

    return hooks, stats


def sample_image_indices(img_start, img_end, knockout_fraction, seed=None):
    """
    Sample a fraction of image token indices to knock out.

    Args:
        img_start: Start index of image tokens
        img_end: End index of image tokens
        knockout_fraction: Fraction of image tokens to block (0.0 to 1.0)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of image token indices to block
    """
    all_img_indices = list(range(img_start, img_end))
    n_to_block = int(len(all_img_indices) * knockout_fraction)

    if seed is not None:
        random.seed(seed)

    # Sample without replacement
    blocked_indices = random.sample(all_img_indices, n_to_block)
    return blocked_indices


def install_last_to_image_block(model, ranges, layer_range=None, knockout_fraction=1.0):
    """
    Install pre-hooks that block image→last token attention path.

    Registers forward pre-hooks on self-attention modules that modify
    the attention_mask to add -inf for (query=last_prompt_token, keys=image_tokens).

    Args:
        model: Qwen3VL model
        ranges: Token ranges dict with "image" and "last" keys
        layer_range: Optional (start, end) tuple for half-open interval [start, end)
        knockout_fraction: Fraction of image tokens to block (0.0 to 1.0)

    Returns:
        Tuple of (hooks_list, stats_dict)
    """
    hooks = []
    stats = {"called": 0, "had_mask": 0, "knockout_fraction": knockout_fraction}

    img_start, img_end = ranges["image"]
    last_pos = ranges["last"]

    # Determine which image indices to block
    if knockout_fraction >= 1.0:
        k_idx = slice(img_start, img_end)  # Block all
    else:
        k_idx = sample_image_indices(img_start, img_end, knockout_fraction)
        stats["n_blocked"] = len(k_idx)
        stats["n_total_img"] = img_end - img_start

    def make_prehook(layer_idx):
        def prehook(module, args, kwargs):
            stats["called"] += 1
            attn_mask = kwargs.get("attention_mask", None)

            if attn_mask is None:
                # Some layers might not have attention_mask in kwargs
                # This can happen with certain optimization modes
                return
            stats["had_mask"] += 1

            # Get sequence length from mask
            q_len = attn_mask.shape[-2]

            # Convert negative index to positive
            last_idx = last_pos if last_pos >= 0 else q_len + last_pos

            # Modify attention mask to block last→image attention
            kwargs["attention_mask"] = add_block_to_attention_mask(
                attn_mask,
                q_idx=last_idx,
                k_idx=k_idx,
            )

        return prehook

    layers = get_language_layers(model)

    for i, layer in enumerate(layers):
        if layer_range is None or (layer_range[0] <= i < layer_range[1]):
            handle = layer.self_attn.register_forward_pre_hook(
                make_prehook(i), with_kwargs=True
            )
            hooks.append(handle)

    return hooks, stats


def install_text_to_image_block(model, ranges, layer_range=None, knockout_fraction=1.0):
    """
    Install pre-hooks that block text→image attention path (mediated path).

    Args:
        model: Qwen3VL model
        ranges: Token ranges dict with "image" and "text" keys
        layer_range: Optional (start, end) tuple for half-open interval [start, end)
        knockout_fraction: Fraction of image tokens to block (0.0 to 1.0)

    Returns:
        Tuple of (hooks_list, stats_dict)
    """
    hooks = []
    stats = {"called": 0, "had_mask": 0, "knockout_fraction": knockout_fraction}

    img_start, img_end = ranges["image"]
    txt_start, txt_end = ranges["text"]

    # Determine which image indices to block
    if knockout_fraction >= 1.0:
        k_idx = slice(img_start, img_end)  # Block all
    else:
        k_idx = sample_image_indices(img_start, img_end, knockout_fraction)
        stats["n_blocked"] = len(k_idx)
        stats["n_total_img"] = img_end - img_start

    def make_prehook(layer_idx):
        def prehook(module, args, kwargs):
            stats["called"] += 1
            attn_mask = kwargs.get("attention_mask", None)

            if attn_mask is None:
                return
            stats["had_mask"] += 1

            # Block all text tokens from attending to (subset of) image tokens
            q_idx = list(range(txt_start, txt_end))

            kwargs["attention_mask"] = add_block_to_attention_mask(
                attn_mask,
                q_idx=q_idx,
                k_idx=k_idx,
            )

        return prehook

    layers = get_language_layers(model)

    for i, layer in enumerate(layers):
        if layer_range is None or (layer_range[0] <= i < layer_range[1]):
            handle = layer.self_attn.register_forward_pre_hook(
                make_prehook(i), with_kwargs=True
            )
            hooks.append(handle)

    return hooks, stats


def normalize_cluster_indices(cluster_spec):
    """
    Convert cluster specification to flat list of token indices.

    Args:
        cluster_spec: Either a single (start, end) tuple or a list of (start, end) tuples

    Returns:
        List of token indices
    """
    if cluster_spec is None:
        return []

    # Check if it's a single tuple (start, end) or list of tuples
    if isinstance(cluster_spec, tuple):
        # Single slice: (start, end)
        return list(range(cluster_spec[0], cluster_spec[1]))
    else:
        # List of slices: [(s1, e1), (s2, e2), ...]
        indices = []
        for start, end in cluster_spec:
            indices.extend(range(start, end))
        return indices


def install_last_to_text_block(model, ranges, layer_range=None):
    """
    Install pre-hooks that block last→text attention path (readout path).

    Blocks the last token from attending to all text tokens.

    Args:
        model: Qwen3VL model
        ranges: Token ranges dict with "text" and "last" keys
        layer_range: Optional (start, end) tuple for half-open interval [start, end)

    Returns:
        Tuple of (hooks_list, stats_dict)
    """
    hooks = []
    stats = {"called": 0, "had_mask": 0}

    txt_start, txt_end = ranges["text"]
    last_pos = ranges["last"]
    k_idx = slice(txt_start, txt_end)

    def make_prehook(layer_idx):
        def prehook(module, args, kwargs):
            stats["called"] += 1
            attn_mask = kwargs.get("attention_mask", None)

            if attn_mask is None:
                return
            stats["had_mask"] += 1

            q_len = attn_mask.shape[-2]
            last_idx = last_pos if last_pos >= 0 else q_len + last_pos

            kwargs["attention_mask"] = add_block_to_attention_mask(
                attn_mask,
                q_idx=last_idx,
                k_idx=k_idx,
            )

        return prehook

    layers = get_language_layers(model)

    for i, layer in enumerate(layers):
        if layer_range is None or (layer_range[0] <= i < layer_range[1]):
            handle = layer.self_attn.register_forward_pre_hook(
                make_prehook(i), with_kwargs=True
            )
            hooks.append(handle)

    return hooks, stats


def install_last_to_cluster_block(model, ranges, cluster_spec, layer_range=None):
    """
    Install pre-hooks that block last→cluster attention path.

    Blocks the last token from attending to specific text token clusters.

    Args:
        model: Qwen3VL model
        ranges: Token ranges dict with "last" key
        cluster_spec: Either (start, end) tuple or list of (start, end) tuples
        layer_range: Optional (start, end) tuple for half-open interval [start, end)

    Returns:
        Tuple of (hooks_list, stats_dict)
    """
    hooks = []

    last_pos = ranges["last"]
    k_indices = normalize_cluster_indices(cluster_spec)

    stats = {"called": 0, "had_mask": 0, "n_cluster_tokens": len(k_indices)}

    def make_prehook(layer_idx):
        def prehook(module, args, kwargs):
            stats["called"] += 1
            attn_mask = kwargs.get("attention_mask", None)

            if attn_mask is None:
                return
            stats["had_mask"] += 1

            q_len = attn_mask.shape[-2]
            last_idx = last_pos if last_pos >= 0 else q_len + last_pos

            kwargs["attention_mask"] = add_block_to_attention_mask(
                attn_mask,
                q_idx=last_idx,
                k_idx=k_indices,
            )

        return prehook

    layers = get_language_layers(model)

    for i, layer in enumerate(layers):
        if layer_range is None or (layer_range[0] <= i < layer_range[1]):
            handle = layer.self_attn.register_forward_pre_hook(
                make_prehook(i), with_kwargs=True
            )
            hooks.append(handle)

    return hooks, stats


def install_cluster_to_image_block(model, ranges, cluster_spec, layer_range=None):
    """
    Install pre-hooks that block cluster→image attention path.

    Blocks specific text token clusters from attending to image tokens.

    Args:
        model: Qwen3VL model
        ranges: Token ranges dict with "image" key
        cluster_spec: Either (start, end) tuple or list of (start, end) tuples
        layer_range: Optional (start, end) tuple for half-open interval [start, end)

    Returns:
        Tuple of (hooks_list, stats_dict)
    """
    hooks = []

    img_start, img_end = ranges["image"]
    k_idx = slice(img_start, img_end)  # Block all image tokens

    # Convert cluster spec to flat list of indices
    q_indices = normalize_cluster_indices(cluster_spec)

    stats = {"called": 0, "had_mask": 0, "n_cluster_tokens": len(q_indices)}

    def make_prehook(layer_idx):
        def prehook(module, args, kwargs):
            stats["called"] += 1
            attn_mask = kwargs.get("attention_mask", None)

            if attn_mask is None:
                return
            stats["had_mask"] += 1

            # Block cluster tokens from attending to image tokens
            kwargs["attention_mask"] = add_block_to_attention_mask(
                attn_mask,
                q_idx=q_indices,
                k_idx=k_idx,
            )

        return prehook

    layers = get_language_layers(model)

    for i, layer in enumerate(layers):
        if layer_range is None or (layer_range[0] <= i < layer_range[1]):
            handle = layer.self_attn.register_forward_pre_hook(
                make_prehook(i), with_kwargs=True
            )
            hooks.append(handle)

    return hooks, stats
