"""Create paired recognition dataset from coco_recognition.

Pairs samples that share the same question but have different answers.
E.g., both ask "What is the person holding?" but one is "bat", other is "book".

Usage:
    python scripts/create_recognition_pairs.py \
        --dataset-path /path/to/datasets \
        --dataset-name coco_recognition
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="coco_recognition")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    ds = load_from_disk(str(dataset_path / f"{args.dataset_name}.hf"))
    print(f"Loaded {len(ds)} samples")

    # Group by question
    question_groups = defaultdict(list)
    for idx in range(len(ds)):
        q = ds[idx]["question_text"]
        question_groups[q].append(idx)

    # Create pairs: within each question group, pair samples with different answers
    samples = []
    pair_id = 0

    for question, indices in question_groups.items():
        if len(indices) < 2:
            continue

        # Get unique answers within this group
        answer_groups = defaultdict(list)
        for idx in indices:
            answer_groups[ds[idx]["preposition"]].append(idx)

        unique_answers = list(answer_groups.keys())
        if len(unique_answers) < 2:
            continue

        # Pair each answer with every other answer (take first sample from each)
        for i in range(len(unique_answers)):
            for j in range(i + 1, len(unique_answers)):
                idx_a = answer_groups[unique_answers[i]][0]
                idx_b = answer_groups[unique_answers[j]][0]

                sample_a = ds[idx_a]
                sample_b = ds[idx_b]

                # Sample A
                samples.append({
                    "image": sample_a["image"],
                    "objects": sample_a["objects"],
                    "preposition": sample_a["preposition"],
                    "question_text": question,
                    "pair_id": pair_id,
                })
                # Sample B
                samples.append({
                    "image": sample_b["image"],
                    "objects": sample_b["objects"],
                    "preposition": sample_b["preposition"],
                    "question_text": question,
                    "pair_id": pair_id,
                })
                pair_id += 1

    print(f"\nCreated {pair_id} pairs ({len(samples)} samples)")

    # Show examples
    print("\nExamples:")
    for i in range(min(10, pair_id)):
        a = samples[i * 2]
        b = samples[i * 2 + 1]
        print(f"  Pair {i}: Q: {a['question_text']}")
        print(f"    A: {a['preposition']:<15} B: {b['preposition']}")

    # Show pair distribution by question
    from collections import Counter
    q_counts = Counter(samples[i * 2]["question_text"] for i in range(pair_id))
    print(f"\nPairs per question:")
    for q, n in q_counts.most_common():
        print(f"  {n:>3} pairs: {q}")

    # Save
    out_ds = Dataset.from_dict({
        "image": [s["image"] for s in samples],
        "objects": [s["objects"] for s in samples],
        "preposition": [s["preposition"] for s in samples],
        "question_text": [s["question_text"] for s in samples],
        "pair_id": [s["pair_id"] for s in samples],
    })

    out_name = f"{args.dataset_name}_pairs"
    out_path = dataset_path / f"{out_name}.hf"
    tmp_path = dataset_path / f"{out_name}_tmp.hf"
    out_ds.save_to_disk(str(tmp_path))
    if out_path.exists():
        shutil.rmtree(str(out_path))
    shutil.move(str(tmp_path), str(out_path))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
