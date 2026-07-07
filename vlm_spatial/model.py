import torch
from transformers import AutoProcessor, AutoConfig


# Map architecture names to their model classes (lazy imports)
_MODEL_CLASS_MAP = {
    "Qwen3VLForConditionalGeneration": "transformers.Qwen3VLForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration": "transformers.LlavaOnevisionForConditionalGeneration",
    "LlavaForConditionalGeneration": "transformers.LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration": "transformers.LlavaNextForConditionalGeneration",
    "InternVLForConditionalGeneration": "transformers.InternVLForConditionalGeneration",
}


def load_model(model_path):
    """
    Load VLM model with eager attention implementation.
    Auto-detects model class from the config at model_path.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model from {model_path}...")

    # Auto-detect model class from config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = config.architectures[0] if config.architectures else None
    class_path = _MODEL_CLASS_MAP.get(arch)

    if class_path is None:
        raise ValueError(
            f"Unsupported model architecture: {arch}. "
            f"Supported: {list(_MODEL_CLASS_MAP.keys())}"
        )

    # Import the model class
    module_name, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)

    print(f"  Architecture: {arch}")
    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    # Suppress "Setting pad_token_id to eos_token_id" warning
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, processor
