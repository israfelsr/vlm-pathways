"""Build the What's-Up–derived datasets into HuggingFace format.

Downloads the What's-Up benchmark (Kamath et al., 2023) from Google Drive and
processes all six splits into the on-disk format used by the experiments:

    controlled_images, controlled_clevr, coco_one, coco_two, vg_qa_one, vg_qa_two

Each output has columns: image, caption_correct, caption_incorrect, preposition, objects.

Usage:
    python data/build_whatsup.py --data-dir ./data
    python data/build_whatsup.py --data-dir ./data --skip-download   # raw already present
"""

import argparse
import json
import re
import tarfile
import zipfile
from pathlib import Path

from datasets import Dataset
from PIL import Image
from tqdm.auto import tqdm


GDRIVE_FOLDER_ID = "164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ"

DATASETS = {
    "controlled_images": {"json": "controlled_images_dataset.json", "archive": "controlled_images.tar.gz", "type": "controlled"},
    "controlled_clevr":  {"json": "controlled_clevr_dataset.json",  "archive": "controlled_clevr.tar.gz",  "type": "controlled"},
    "coco_one":          {"json": "coco_qa_one_obj.json",           "archive": "val2017.zip",              "type": "coco"},
    "coco_two":          {"json": "coco_qa_two_obj.json",           "archive": "val2017.zip",              "type": "coco"},
    "vg_qa_one":         {"json": "vg_qa_one_obj.json",             "archive": "vg_images.tar.gz",         "type": "vg"},
    "vg_qa_two":         {"json": "vg_qa_two_obj.json",             "archive": "vg_images.tar.gz",         "type": "vg"},
}


# --- Caption / filename parsing ---------------------------------------------

def extract_preposition(caption):
    """Extract preposition from a caption (coco_one / vg_qa_one legacy path)."""
    caption_lower = caption.lower()
    patterns = [
        r"on the (top|bottom|left|right)",
        r"to the (left|right|front|behind) of",
        r"\s(on|under|above|below|behind|left|right|front)\s",
    ]
    for pattern in patterns:
        match = re.search(pattern, caption_lower)
        if match:
            return match.group(1)
    return "unknown"


def extract_objects(caption):
    """Extract [obj1, obj2] from a caption (coco_one / vg_qa_one legacy path)."""
    caption = re.sub(r"^A photo of ", "", caption, flags=re.IGNORECASE)
    prep_patterns = [
        r" to the (?:left|right|front|behind) of ",
        r" on the (?:top|bottom|left|right) ",
        r" (?:on|under|above|below|behind) ",
    ]
    for pattern in prep_patterns:
        parts = re.split(pattern, caption, flags=re.IGNORECASE)
        if len(parts) == 2:
            obj1 = re.sub(r"^(a|an|the)\s+", "", parts[0].strip(), flags=re.IGNORECASE)
            obj2 = re.sub(r"^(a|an|the)\s+", "", parts[1].strip(), flags=re.IGNORECASE)
            return [obj1, obj2]
    words = caption.split()
    if len(words) >= 2:
        obj1 = re.sub(r"^(a|an|the)$", "", words[0], flags=re.IGNORECASE)
        obj2 = re.sub(r"^(a|an|the)$", "", words[-1], flags=re.IGNORECASE)
        return [obj1 or "unknown", obj2 or "unknown"]
    return [caption, "unknown"]


def extract_from_filename_controlled(image_path):
    """Parse controlled filenames: {object1}_{preposition}_{object2}.jpeg."""
    stem = Path(image_path).stem
    parts = stem.split("_")
    obj1 = parts[0].replace("-", " ")
    obj2 = parts[-1].replace("-", " ")
    prep_raw = "_".join(parts[1:-1])
    prep_map = {
        "right_of": "right", "left_of": "left", "in-front_of": "front",
        "behind": "behind", "on": "on", "under": "under",
    }
    return obj1, prep_map.get(prep_raw, prep_raw), obj2


def extract_from_caption_photo(caption):
    """Parse 'A photo of a {obj1} {preposition} a {obj2}' (coco_two / vg_qa_two)."""
    pattern = (r"^A photo of an? (.+?) "
               r"(to the left of|to the right of|to the front of|to the behind of|above|below) "
               r"an? (.+)$")
    match = re.match(pattern, caption, re.IGNORECASE)
    if match:
        prep_map = {
            "to the left of": "left", "to the right of": "right",
            "to the front of": "front", "to the behind of": "behind",
            "above": "above", "below": "below",
        }
        return match.group(1).strip(), prep_map.get(match.group(2).strip().lower(), match.group(2)), match.group(3).strip()
    return "unknown", "unknown", "unknown"


# --- Per-type processors -----------------------------------------------------

def process_controlled(name, config, download_dir):
    with open(download_dir / config["json"]) as f:
        data_raw = json.load(f)
    data = {k: [] for k in ("image", "caption_correct", "caption_incorrect", "preposition", "objects")}
    for sample in tqdm(data_raw, desc=name):
        image_path_str = sample["image_path"]
        if image_path_str.startswith("data/"):
            image_path_str = image_path_str[5:]
        image_path = download_dir / image_path_str
        if not image_path.exists():
            print(f"Warning: image not found: {image_path}")
            continue
        obj1, prep, obj2 = extract_from_filename_controlled(image_path_str)
        data["image"].append(Image.open(image_path).convert("RGB"))
        data["caption_correct"].append(sample["caption_options"][0])
        data["caption_incorrect"].append(sample["caption_options"][1:])
        data["preposition"].append(prep)
        data["objects"].append([obj1, obj2])
    return Dataset.from_dict(data)


def process_coco(name, config, download_dir):
    with open(download_dir / config["json"]) as f:
        data_raw = json.load(f)
    data = {k: [] for k in ("image", "caption_correct", "caption_incorrect", "preposition", "objects")}
    for sample in tqdm(data_raw, desc=name):
        image_id, caption_correct, caption_incorrect = sample[0], sample[1], sample[2]
        image_path = download_dir / "val2017" / f"{image_id:012d}.jpg"
        if not image_path.exists():
            print(f"Warning: image not found: {image_path}")
            continue
        if name == "coco_two":
            obj1, prep, obj2 = extract_from_caption_photo(caption_correct)
            objects = [obj1, obj2]
        else:
            prep, objects = extract_preposition(caption_correct), extract_objects(caption_correct)
        data["image"].append(Image.open(image_path).convert("RGB"))
        data["caption_correct"].append(caption_correct)
        data["caption_incorrect"].append([caption_incorrect])
        data["preposition"].append(prep)
        data["objects"].append(objects)
    return Dataset.from_dict(data)


def process_vg(name, config, download_dir):
    with open(download_dir / config["json"]) as f:
        data_raw = json.load(f)
    data = {k: [] for k in ("image", "caption_correct", "caption_incorrect", "preposition", "objects")}
    skipped = 0
    for sample in tqdm(data_raw, desc=name):
        image_id, caption_correct, caption_incorrect = sample[0], sample[1], sample[2]
        image_path = next(download_dir.rglob(f"{image_id}.jpg"), None)
        if image_path is None or not image_path.exists():
            print(f"Warning: image not found: {image_id}.jpg")
            continue
        if name == "vg_qa_two":
            obj1, prep, obj2 = extract_from_caption_photo(caption_correct)
            objects = [obj1, obj2]
            if prep == "unknown":
                skipped += 1
                continue
        else:
            prep, objects = extract_preposition(caption_correct), extract_objects(caption_correct)
        data["image"].append(Image.open(image_path).convert("RGB"))
        data["caption_correct"].append(caption_correct)
        data["caption_incorrect"].append([caption_incorrect])
        data["preposition"].append(prep)
        data["objects"].append(objects)
    if skipped:
        print(f"  Skipped {skipped} samples with unknown preposition")
    return Dataset.from_dict(data)


PROCESSORS = {"controlled": process_controlled, "coco": process_coco, "vg": process_vg}


def main():
    parser = argparse.ArgumentParser(description="Build What's-Up datasets")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Output dir (and where raw data is downloaded/extracted)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the Google Drive download (raw data already present)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    download_dir = data_dir / "whatsup_raw"
    download_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        import gdown
        print("Downloading What's-Up benchmark from Google Drive...")
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(download_dir), quiet=False)

    # Extract each archive once
    extracted = set()
    for config in DATASETS.values():
        archive = config["archive"]
        if archive in extracted:
            continue
        archive_path = download_dir / archive
        if not archive_path.exists():
            continue  # already extracted on a previous run
        print(f"Extracting {archive}...")
        if archive.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=download_dir)
        elif archive.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(path=download_dir)
        extracted.add(archive)
        archive_path.unlink()

    # Process and save
    for name, config in DATASETS.items():
        print(f"\n{'='*60}\nProcessing {name}\n{'='*60}")
        dataset = PROCESSORS[config["type"]](name, config, download_dir)
        print(f"{len(dataset)} samples | prepositions: {sorted(set(dataset['preposition']))}")
        out = data_dir / f"{name}.hf"
        dataset.save_to_disk(str(out))
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
