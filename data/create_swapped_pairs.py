"""Create object-swapped version of shapes pairs dataset.

Original: "Where is obj1 wrt obj2?" → "left"  (obj1 is left of center obj2)
Swapped:  "Where is obj2 wrt obj1?" → "right" (center obj2 is right of peripheral obj1)

Same images, swapped question objects, flipped preposition.
Tests global vs local spatial reasoning bias.

Usage:
    python scripts/create_swapped_pairs.py --dataset-path /path/to/datasets \
        --dataset-name controlled_shapes_pairs
"""

import argparse
import shutil
from pathlib import Path

from datasets import Dataset, load_from_disk


FLIP_MAP = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


def main():
    parser = argparse.ArgumentParser(description="Create object-swapped pairs dataset")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="controlled_shapes_pairs")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    ds = load_from_disk(str(dataset_path / f"{args.dataset_name}.hf"))
    print(f"Loaded {len(ds)} samples from {args.dataset_name}")
    print(f"Prepositions: {set(ds['preposition'])}")

    swapped_objects = []
    swapped_preps = []
    skipped = 0

    for idx in range(len(ds)):
        sample = ds[idx]
        prep = sample["preposition"]
        objects = sample["objects"]

        if prep not in FLIP_MAP:
            skipped += 1
            continue

        # Swap objects: [obj1, obj2] → [obj2, obj1]
        swapped_objects.append([objects[1], objects[0]])
        swapped_preps.append(FLIP_MAP[prep])

    if skipped > 0:
        print(f"Skipped {skipped} samples with unmapped prepositions")

    # Keep only the samples we didn't skip
    keep_indices = [i for i in range(len(ds)) if ds[i]["preposition"] in FLIP_MAP]
    ds_filtered = ds.select(keep_indices)

    out_ds = Dataset.from_dict({
        "image": ds_filtered["image"],
        "objects": swapped_objects,
        "preposition": swapped_preps,
        "pair_id": ds_filtered["pair_id"],
    })

    print(f"\nSwapped dataset: {len(out_ds)} samples")
    print(f"Prepositions: {set(out_ds['preposition'])}")

    # Show examples
    for i in range(min(5, len(out_ds))):
        orig = ds[keep_indices[i]]
        swap = out_ds[i]
        print(f"  Original: {orig['objects']} → {orig['preposition']}")
        print(f"  Swapped:  {swap['objects']} → {swap['preposition']}")
        print()

    out_name = f"{args.dataset_name}_swapped"
    out_path = dataset_path / f"{out_name}.hf"
    tmp_path = dataset_path / f"{out_name}_tmp.hf"
    out_ds.save_to_disk(str(tmp_path))
    if out_path.exists():
        shutil.rmtree(str(out_path))
    shutil.move(str(tmp_path), str(out_path))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
