"""Convert VSR dataset to our pipeline format.

Filters to clean directional spatial relations, groups synonyms,
and outputs in the same format as our other datasets:
  - image: PIL Image
  - objects: [subj, obj]
  - preposition: one of {above, below, left, right, behind, front}

Only keeps label=1 samples (relation is true for the image).

Usage:
    python data/convert_vsr.py \
        --input ./data/vsr_raw.hf \
        --output ./data/vsr_spatial.hf
"""

import argparse
from collections import Counter

from datasets import load_from_disk, Dataset

# Map VSR relations to clean directional labels
RELATION_MAP = {
    # above
    "above": "above",
    "on top of": "above",
    "on": "above",
    "over": "above",
    # below
    "below": "below",
    "under": "below",
    "beneath": "below",
    # left
    "left of": "left",
    "at the left side of": "left",
    # right
    "right of": "right",
    "at the right side of": "right",
    # behind
    "behind": "behind",
    "at the back of": "behind",
    # front
    "in front of": "front",
}


def main():
    parser = argparse.ArgumentParser(description="Convert VSR to spatial pipeline format")
    parser.add_argument("--input", type=str, required=True, help="Path to VSR .hf dataset")
    parser.add_argument("--output", type=str, required=True, help="Output path for converted .hf")
    parser.add_argument("--preview", type=int, default=0, help="Number of preview images to save")
    parser.add_argument("--max-size", type=int, default=448,
                        help="Max image dimension in pixels (default: 448). "
                             "Images larger than this are resized preserving aspect ratio.")
    args = parser.parse_args()

    dataset = load_from_disk(args.input)
    print(f"Loaded VSR: {len(dataset)} samples")

    # Filter: label=1 AND relation in our map
    samples = []
    skipped_label = 0
    skipped_relation = 0

    for i in range(len(dataset)):
        s = dataset[i]
        if s["label"] != 1:
            skipped_label += 1
            continue
        mapped = RELATION_MAP.get(s["relation"])
        if mapped is None:
            skipped_relation += 1
            continue
        samples.append({
            "image": s["image"],
            "objects": [s["subj"], s["obj"]],
            "preposition": mapped,
        })

    print(f"\nFiltered: {len(samples)} spatial samples")
    print(f"  Skipped (label=0): {skipped_label}")
    print(f"  Skipped (non-spatial relation): {skipped_relation}")

    # Distribution
    counts = Counter(s["preposition"] for s in samples)
    print(f"\nDistribution:")
    for p in sorted(counts):
        print(f"  {p}: {counts[p]}")

    # Save
    out_dataset = Dataset.from_dict({
        "image": [s["image"] for s in samples],
        "objects": [s["objects"] for s in samples],
        "preposition": [s["preposition"] for s in samples],
    })
    out_dataset.save_to_disk(args.output)
    print(f"\nSaved to {args.output}")

    # Preview
    if args.preview > 0:
        from pathlib import Path
        preview_dir = Path(args.output).parent / "previews" / "vsr_spatial"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(samples[:args.preview]):
            obj_str = "_".join(o.replace(" ", "_") for o in s["objects"])
            fname = f"{i:03d}_{obj_str}_{s['preposition']}.png"
            s["image"].save(preview_dir / fname)
            print(f"  Preview: {fname}")


if __name__ == "__main__":
    main()
