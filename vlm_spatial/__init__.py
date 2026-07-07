"""VLM Spatial Mechanisms - shared utilities for attention analysis experiments."""

from vlm_spatial.model import load_model
from vlm_spatial.data import load_dataset, create_question, find_token_ranges, find_cluster_ranges
from vlm_spatial.hooks import (
    get_language_layers,
    add_block_to_attention_mask,
    add_boost_to_attention_mask,
    sample_image_indices,
    install_last_to_image_block,
    install_text_to_image_block,
    install_cluster_to_image_block,
    install_last_to_text_block,
    install_last_to_cluster_block,
    install_attention_boost,
    normalize_cluster_indices,
)
from vlm_spatial.config import load_config, build_dataset_path, parse_layer_range

__all__ = [
    # Model
    "load_model",
    # Data
    "load_dataset",
    "create_question",
    "find_token_ranges",
    "find_cluster_ranges",
    # Hooks
    "get_language_layers",
    "add_block_to_attention_mask",
    "add_boost_to_attention_mask",
    "sample_image_indices",
    "install_last_to_image_block",
    "install_text_to_image_block",
    "install_cluster_to_image_block",
    "install_last_to_text_block",
    "install_last_to_cluster_block",
    "install_attention_boost",
    "normalize_cluster_indices",
    # Config
    "load_config",
    "build_dataset_path",
    "parse_layer_range",
]
