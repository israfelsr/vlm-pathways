"""
    python experiments/baseline_recognition_pairs.py --config configs/default.yaml \
        --dataset-name coco_recognition_pairs --results-subdir qwen/baseline_open
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from vlm_spatial.config import load_config
from vlm_spatial.data import load_dataset
from vlm_spatial.model import load_model


SYNONYMS = {
    "bird": [
        "bird", "pigeon", "seagull", "turkey", "parrot", "sparrow",
        "crow", "eagle", "duck", "goose", "owl", "penguin",
        "hen", "rooster", "chicken",
    ],
    "hair drier": ["hair drier", "hair dryer", "hairdryer", "blow dryer"],
    "cell phone": ["cell phone", "cellphone", "phone", "smartphone", "mobile"],
    "remote": ["remote", "remote control"],
    "hot dog": ["hot dog", "hotdog"],
    "baseball bat": ["baseball bat", "bat"],
    "cup": ["cup", "mug"],
    "person": ["person", "man", "woman", "child", "boy", "girl", "kid"],
    "giraffe": ["giraffe"],
    "dog": ["dog", "puppy"],
    "cat": ["cat", "kitten"],
    "bicycle": ["bicycle", "bike"],
    "laptop": ["laptop", "computer", "notebook"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="coco_recognition_pairs")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--results-subdir", type=str, default="baseline_recognition")
    args = parser.parse_args()

    config = load_config(args.config)
    model, processor = load_model(config["model_path"])
    dataset, prepositions = load_dataset(config["dataset_path"], args.dataset_name)

    n_samples = min(args.n_samples or len(dataset), len(dataset))

    # Build pair lookup: pair_id -> [idx_a, idx_b]
    pair_map = {}
    for idx in range(len(dataset)):
        pid = dataset[idx]["pair_id"]
        if pid not in pair_map:
            pair_map[pid] = []
        pair_map[pid].append(idx)

    # Get all unique answers and their first token IDs
    all_answers = sorted(set(dataset["preposition"]))
    answer_first_tokens = {}
    for ans in all_answers:
        token_ids = processor.tokenizer.encode(ans, add_special_tokens=False)
        answer_first_tokens[ans] = token_ids[0]
    print(f"Answer categories: {len(all_answers)}")
    for ans in all_answers:
        tok = processor.tokenizer.decode([answer_first_tokens[ans]])
        print(f"  {ans:<20} -> first token: '{tok}' (id={answer_first_tokens[ans]})")

    # Check for first-token collisions
    token_to_answers = {}
    for ans, tid in answer_first_tokens.items():
        if tid not in token_to_answers:
            token_to_answers[tid] = []
        token_to_answers[tid].append(ans)
    collisions = {tid: anss for tid, anss in token_to_answers.items() if len(anss) > 1}
    if collisions:
        print(f"\nWARNING: First-token collisions detected:")
        for tid, anss in collisions.items():
            tok = processor.tokenizer.decode([tid])
            print(f"  token '{tok}' (id={tid}): {anss}")

    results = []
    correct_gen = 0
    correct_constrained = 0

    for idx in tqdm(range(n_samples), desc="Baseline"):
        sample = dataset[idx]
        image = sample["image"]
        gt = sample["preposition"]
        question = sample.get("question_text", "What is in the image?")
        pair_id = sample["pair_id"]

        # Find pair's GT
        pair_indices = pair_map[pair_id]
        pair_idx = [i for i in pair_indices if i != idx][0]
        pair_gt = dataset[pair_idx]["preposition"]

        # Prepare inputs with prefill "It is a"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is a"}],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True,
        )
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)

        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Generated text
        generated = processor.tokenizer.decode(
            output.sequences[0, prompt_len:], skip_special_tokens=True
        ).strip()

        # First-token logits and probabilities
        first_logits = output.scores[0][0]  # [vocab_size]
        first_probs = torch.softmax(first_logits, dim=-1)

        # P(first token of each answer category)
        option_probs = {}
        option_logits = {}
        for ans in all_answers:
            tid = answer_first_tokens[ans]
            option_probs[ans] = first_probs[tid].item()
            option_logits[ans] = first_logits[tid].item()

        # Constrained argmax over answer categories
        constrained_choice = max(option_logits, key=option_logits.get)
        is_correct_constrained = constrained_choice == gt
        correct_constrained += int(is_correct_constrained)

        # Generate-based matching (with synonyms)
        variants = SYNONYMS.get(gt, [gt])
        has_answer = any(
            re.search(rf"\b{re.escape(v)}\b", generated, re.IGNORECASE)
            for v in variants
        )
        correct_gen += int(has_answer)

        results.append({
            "idx": idx,
            "pair_id": pair_id,
            "question": question,
            "ground_truth": gt,
            "pair_ground_truth": pair_gt,
            "generated": generated,
            "correct_generate": has_answer,
            "correct_constrained": is_correct_constrained,
            "constrained_choice": constrained_choice,
            "p_gt": option_probs[gt],
            "p_pair_gt": option_probs[pair_gt],
            "option_probs": {k: round(v, 6) for k, v in option_probs.items()},
            "option_logits": {k: round(v, 4) for k, v in option_logits.items()},
        })

        if idx < 10 or not has_answer:
            mark = "OK" if has_answer else "MISS"
            c_mark = "OK" if is_correct_constrained else "MISS"
            print(f"  [{mark}|{c_mark}] Q: {question}")
            print(f"       GT: {gt:<20} pair_GT: {pair_gt}")
            print(f"       Gen: It is a {generated[:60]}")
            print(f"       Constrained: {constrained_choice}")
            print(f"       P(gt)={option_probs[gt]:.4f}  P(pair_gt)={option_probs[pair_gt]:.4f}")

    print(f"\n{'='*60}")
    print(f"Generate accuracy:    {correct_gen}/{n_samples} ({correct_gen/n_samples*100:.1f}%)")
    print(f"Constrained accuracy: {correct_constrained}/{n_samples} ({correct_constrained/n_samples*100:.1f}%)")
    print(f"{'='*60}")

    # Per-answer breakdown
    print(f"\nPer-answer accuracy (constrained):")
    answer_correct = Counter()
    answer_total = Counter()
    for r in results:
        answer_total[r["ground_truth"]] += 1
        if r["correct_constrained"]:
            answer_correct[r["ground_truth"]] += 1
    for ans in sorted(answer_total.keys()):
        c = answer_correct[ans]
        t = answer_total[ans]
        print(f"  {ans:<20} {c}/{t} ({c/t*100:.0f}%)")

    # Save
    output_dir = Path("results") / args.results_subdir / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "accuracy_generate": correct_gen / n_samples,
        "accuracy_constrained": correct_constrained / n_samples * 100,
        "accuracy_top1": correct_constrained / n_samples * 100,
        "n_samples": n_samples,
        "n_answers": len(all_answers),
        "answer_first_tokens": {k: int(v) for k, v in answer_first_tokens.items()},
        "samples": results,
    }
    json_path = output_dir / "baseline_samples.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {json_path}")

    # Also save summary compatible with consolidate_results.py
    summary = {
        "accuracy_top1": correct_constrained / n_samples * 100,
        "accuracy_constrained": correct_constrained / n_samples * 100,
        "n_samples": n_samples,
    }
    with open(output_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
