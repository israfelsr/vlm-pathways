"""Create mirrored pair datasets from COCO/VG by horizontal flip (left↔right only).

For each sample with preposition "left" or "right":
  - Sample A (even index): original image, original label
  - Sample B (odd index): horizontally flipped image, flipped label
  - Both share the same pair_id

Usage:
    python scripts/create_mirror_pairs.py --dataset-path /path/to/datasets \
        --datasets coco_two vg_qa_two coco_one vg_qa_one
"""

import argparse
import shutil
from pathlib import Path

from datasets import Dataset, load_from_disk
from PIL import Image


HFLIP_MAP = {"left": "right", "right": "left"}


def create_hflip_pairs(dataset, ds_name):
    samples = []
    pair_id = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        prep = sample["preposition"]
        if prep not in HFLIP_MAP:
            continue

        flipped_image = sample["image"].transpose(Image.FLIP_LEFT_RIGHT)
        flipped_prep = HFLIP_MAP[prep]

        # Sample A: original
        samples.append({
            "image": sample["image"],
            "objects": sample["objects"],
            "preposition": prep,
            "pair_id": pair_id,
        })
        # Sample B: flipped
        samples.append({
            "image": flipped_image,
            "objects": sample["objects"],
            "preposition": flipped_prep,
            "pair_id": pair_id,
        })
        pair_id += 1

    if not samples:
        print(f"  {ds_name}_hflip: no left/right samples found")
        return None

    ds = Dataset.from_dict({
        "image": [s["image"] for s in samples],
        "objects": [s["objects"] for s in samples],
        "preposition": [s["preposition"] for s in samples],
        "pair_id": [s["pair_id"] for s in samples],
    })
    print(f"  {ds_name}_hflip: {pair_id} pairs ({len(ds)} samples), "
          f"preps: {set(ds['preposition'])}")
    return ds


def main():
    parser = argparse.ArgumentParser(description="Create mirrored pair datasets (hflip, left↔right)")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True,
                        default=["coco_two", "vg_qa_two", "coco_one", "vg_qa_one"])
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    for ds_name in args.datasets:
        src_path = dataset_path / f"{ds_name}.hf"
        print(f"\nLoading {src_path}...")
        dataset = load_from_disk(str(src_path))
        print(f"  {len(dataset)} samples, prepositions: {set(dataset['preposition'])}")

        ds = create_hflip_pairs(dataset, ds_name)
        if ds is None:
            continue

        out_name = f"{ds_name}_hflip"
        out_path = dataset_path / f"{out_name}.hf"
        tmp_path = dataset_path / f"{out_name}_tmp.hf"

        ds.save_to_disk(str(tmp_path))
        if out_path.exists():
            shutil.rmtree(str(out_path))
        shutil.move(str(tmp_path), str(out_path))
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
