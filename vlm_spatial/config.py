from pathlib import Path

import yaml


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict with configuration values
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")
    return config


def build_dataset_path(config):
    if "dataset_path" in config and "dataset_name" in config:
        dataset_base = Path(config["dataset_path"])
        dataset_name = config["dataset_name"]
        return str(dataset_base / f"{dataset_name}.hf")
    elif "dataset" in config:
        return config["dataset"]
    else:
        raise ValueError("Config must have either 'dataset_path'+'dataset_name' or 'dataset'")


def parse_layer_range(layers_str):
    if layers_str is None or layers_str == "all":
        return None

    start, end = map(int, layers_str.split("-"))
    return (start, end)
