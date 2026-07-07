"""
    python experiments/baseline_recognition_fc.py --config configs/default.yaml \
        --dataset-name coco_recognition_pairs --results-subdir qwen/baseline
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from vlm_spatial.config import load_config
from vlm_spatial.data import load_dataset
from vlm_spatial.model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="coco_recognition_pairs")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--results-subdir", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    model, processor = load_model(config["model_path"])
    dataset, prepositions = load_dataset(config["dataset_path"], args.dataset_name,
                                         task_type="recognition")

    all_answers = sorted(prepositions)
    n_samples = min(args.n_samples or len(dataset), len(dataset))

    # Build pair lookup
    pair_map = {}
    for idx in range(len(dataset)):
        pid = dataset[idx]["pair_id"]
        if pid not in pair_map:
            pair_map[pid] = []
        pair_map[pid].append(idx)

    # Build per-sample option sets (seeded for reproducibility)
    rng = random.Random(args.seed)
    sample_options = {}
    for pid, indices in pair_map.items():
        if len(indices) == 2:
            gt_a = dataset[indices[0]]["preposition"]
            gt_b = dataset[indices[1]]["preposition"]
            distractors = [a for a in all_answers if a not in (gt_a, gt_b)]
            chosen = rng.sample(distractors, min(2, len(distractors)))
            options = sorted([gt_a, gt_b] + chosen)
            for idx in indices:
                sample_options[idx] = options
        else:
            for idx in indices:
                gt = dataset[idx]["preposition"]
                distractors = [a for a in all_answers if a != gt]
                chosen = rng.sample(distractors, min(3, len(distractors)))
                sample_options[idx] = sorted([gt] + chosen)

    # Build first-token IDs for all answers
    answer_token_ids = {}
    for ans in all_answers:
        token_ids = processor.tokenizer.encode(ans, add_special_tokens=False)
        answer_token_ids[ans] = token_ids[0]

    results = []
    correct_constrained = 0
    correct_fc = 0

    for idx in tqdm(range(n_samples), desc="Baseline FC"):
        sample = dataset[idx]
        image = sample["image"]
        gt = sample["preposition"]
        base_question = sample.get("question_text", "What is in the image?")
        options = sample_options[idx]

        # Build forced-choice question
        question = f"{base_question} Answer only with {', '.join(options)}."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        first_logits = output.scores[0][0]

        # Constrained over the 4 FC options
        fc_logits = {ans: first_logits[answer_token_ids[ans]].item() for ans in options}
        fc_choice = max(fc_logits, key=fc_logits.get)
        is_correct_fc = fc_choice == gt
        correct_fc += int(is_correct_fc)

        # Also constrained over all 22 (for comparison)
        all_logits = {ans: first_logits[answer_token_ids[ans]].item() for ans in all_answers}
        all_choice = max(all_logits, key=all_logits.get)
        is_correct_all = all_choice == gt
        correct_constrained += int(is_correct_all)

        results.append({
            "idx": idx,
            "pair_id": sample.get("pair_id"),
            "question": question,
            "ground_truth": gt,
            "options": options,
            "fc_choice": fc_choice,
            "correct_fc": is_correct_fc,
            "correct_constrained": is_correct_all,
            "constrained_choice": all_choice,
            "option_logits_fc": {k: round(v, 4) for k, v in fc_logits.items()},
            "option_logits_all": {k: round(v, 4) for k, v in all_logits.items()},
        })

        if idx < 5:
            mark = "OK" if is_correct_fc else "MISS"
            print(f"  [{mark}] Q: {question[:70]}")
            print(f"       GT: {gt:<15} FC: {fc_choice:<15} All22: {all_choice}")

    print(f"\n{'='*60}")
    print(f"FC accuracy (4 options):  {correct_fc}/{n_samples} ({correct_fc/n_samples*100:.1f}%)")
    print(f"Constrained (22 options): {correct_constrained}/{n_samples} ({correct_constrained/n_samples*100:.1f}%)")
    print(f"{'='*60}")

    # Save
    output_dir = Path("results") / args.results_subdir / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "accuracy_fc": correct_fc / n_samples * 100,
        "accuracy_constrained": correct_constrained / n_samples * 100,
        "accuracy_top1": correct_fc / n_samples * 100,
        "n_samples": n_samples,
        "n_options": 4,
        "seed": args.seed,
        "samples": results,
    }
    with open(output_dir / "baseline_samples.json", "w") as f:
        json.dump(out, f, indent=2)

    summary = {
        "accuracy_top1": correct_fc / n_samples * 100,
        "accuracy_constrained": correct_fc / n_samples * 100,
        "n_samples": n_samples,
    }
    with open(output_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
