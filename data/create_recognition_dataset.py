"""Create a recognition HF dataset from the llava-interp clean_questions.json + COCO.

Each sample has:
  - image: the COCO image
  - objects: [object_name] (the target object category)
  - preposition: the answer (object category name, for compatibility with our pipeline)
  - question: the manually curated question (e.g., "What is on the table?")

Usage:
    # First download clean_questions.json:
    # wget https://raw.githubusercontent.com/clemneo/llava-interp/main/data/clean_questions.json \
    #     -O dataset/clean_questions.json

    python scripts/create_recognition_dataset.py \
        --coco-images /path/to/train2017 \
        --coco-annotations /path/to/annotations/instances_train2017.json \
        --questions /path/to/clean_questions.json \
        --output /path/to/dataset/coco_recognition.hf
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from PIL import Image
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser(description="Create recognition dataset from COCO + llava-interp questions")
    parser.add_argument("--coco-images", type=str, required=True, help="Path to train2017/ directory")
    parser.add_argument("--coco-annotations", type=str, required=True, help="Path to instances_train2017.json")
    parser.add_argument("--questions", type=str, required=True, help="Path to clean_questions.json")
    parser.add_argument("--output", type=str, required=True, help="Output path for HF dataset")
    args = parser.parse_args()

    # Load COCO annotations
    print("Loading COCO annotations...")
    coco = COCO(args.coco_annotations)

    # Load questions
    with open(args.questions) as f:
        questions_dict = json.load(f)

    # Filter to entries with actual questions
    entries_with_questions = {k: v for k, v in questions_dict.items() if v}
    print(f"Found {len(entries_with_questions)} images with questions")

    # Build dataset
    samples = []
    skipped = 0

    for img_id_str, questions in entries_with_questions.items():
        img_id = int(img_id_str)

        # Load image
        img_info = coco.loadImgs(img_id)
        if not img_info:
            print(f"  Image {img_id} not found in COCO, skipping")
            skipped += 1
            continue

        img_path = Path(args.coco_images) / img_info[0]["file_name"]
        if not img_path.exists():
            print(f"  Image file {img_path} not found, skipping")
            skipped += 1
            continue

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            print(f"  No annotations for image {img_id}, skipping")
            skipped += 1
            continue

        # Find the target object using the paper's filter:
        # exactly 1 instance of a category, area 1000-2000
        target_ann = None
        category_groups = {}
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in category_groups:
                category_groups[cat_id] = []
            category_groups[cat_id].append(ann)

        for cat_id, group in category_groups.items():
            if len(group) == 1 and 1000 < group[0]["area"] < 2000:
                target_ann = group[0]
                break

        # If strict filter doesn't find one, use the largest single-instance object
        if target_ann is None:
            for cat_id, group in category_groups.items():
                if len(group) == 1:
                    if target_ann is None or group[0]["area"] > target_ann["area"]:
                        target_ann = group[0]

        if target_ann is None:
            print(f"  No suitable target object for image {img_id}, skipping")
            skipped += 1
            continue

        cat_info = coco.loadCats(target_ann["category_id"])[0]
        object_name = cat_info["name"]

        img = Image.open(img_path).convert("RGB")

        for question in questions:
            samples.append({
                "image": img,
                "objects": [object_name],
                "preposition": object_name,  # answer = object name
                "question_text": question,
                "image_id": img_id,
            })

    print(f"\nCreated {len(samples)} samples from {len(entries_with_questions) - skipped} images")
    print(f"Skipped {skipped} images")

    if not samples:
        print("No samples created!")
        return

    # Print some examples
    print("\nExamples:")
    for s in samples[:5]:
        print(f"  Q: {s['question_text']:<35} A: {s['preposition']:<15} "
              f"img_id: {s['image_id']}")

    # Show answer distribution
    from collections import Counter
    answer_counts = Counter(s["preposition"] for s in samples)
    print(f"\nAnswer distribution ({len(answer_counts)} unique):")
    for ans, cnt in answer_counts.most_common(15):
        print(f"  {ans:<20}: {cnt}")

    # Save as HF dataset
    ds = Dataset.from_dict({
        "image": [s["image"] for s in samples],
        "objects": [s["objects"] for s in samples],
        "preposition": [s["preposition"] for s in samples],
        "question_text": [s["question_text"] for s in samples],
        "image_id": [s["image_id"] for s in samples],
    })

    ds.save_to_disk(args.output)
    print(f"\nSaved to {args.output}")
    print(f"Dataset: {ds}")


if __name__ == "__main__":
    main()
