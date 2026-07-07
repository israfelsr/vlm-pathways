"""
Usage (from repo root):
    python experiments/02_attention_knockout/baseline.py --config configs/default.yaml
    python experiments/02_attention_knockout/baseline.py --config configs/default.yaml --n-samples 100
"""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

from vlm_spatial import load_model, load_dataset, create_question, load_config


def evaluate_sample_baseline(
    model, processor, image, question, ground_truth, objects,
    no_image=False, question_format="forced_choice", prepositions=None,
):
    """
    Evaluate a single sample using model.generate() with output_scores.

    Returns detailed baseline info comparing generate vs argmax.
    For "free" format, generates up to 20 tokens and uses regex matching.
    """
    # Prepare inputs
    if no_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            }
        ]
    else:
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

    if no_image:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    prompt_len = inputs.input_ids.shape[1]
    max_new_tokens = 20 if question_format == "free" else 1

    with torch.no_grad():
        # Generate with scores
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    # Get generated sequence (excluding prompt)
    generated_ids = output.sequences[0, prompt_len:]
    generated_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # Get first generated token
    first_generated_id = generated_ids[0].item()
    first_generated_token = processor.tokenizer.decode(
        [first_generated_id], skip_special_tokens=True
    ).strip()

    # Get scores for first token and compute argmax
    first_token_scores = output.scores[0]  # [batch, vocab]
    argmax_id = first_token_scores.argmax(dim=-1).item()
    argmax_token = processor.tokenizer.decode(
        [argmax_id], skip_special_tokens=True
    ).strip()
    max_logit = first_token_scores[0, argmax_id].item()

    # Check if generate and argmax match
    generate_argmax_match = first_generated_id == argmax_id

    # Get top-5 tokens from scores
    top5_values, top5_ids = first_token_scores[0].topk(5)
    top5_tokens = [
        processor.tokenizer.decode([tid], skip_special_tokens=True).strip()
        for tid in top5_ids.tolist()
    ]
    top5_logits = top5_values.tolist()

    # Check correctness using first token ID match
    # This handles multi-token GT like "behind" = "beh" + "ind"
    gt_token_ids = processor.tokenizer.encode(ground_truth, add_special_tokens=False)
    gt_first_token_id = gt_token_ids[0]
    correct_first_token_id = first_generated_id == gt_first_token_id

    # Fallback: compare decoded text (handles " red" vs "red" token variants)
    if not correct_first_token_id:
        correct_first_token_id = first_generated_token.lower() == ground_truth.lower()

    # GT first token in top-5 (by token ID, with text fallback)
    top5_ids_list = top5_ids.tolist()
    correct_top5_token_id = gt_first_token_id in top5_ids_list
    if not correct_top5_token_id:
        correct_top5_token_id = any(
            t.lower() == ground_truth.lower() for t in top5_tokens
        )

    # Option-constrained logits: for single-token evaluation, save the logit
    # for each valid answer option. This allows computing accuracy by taking
    # argmax over only the valid options (open prompt, constrained evaluation).
    option_logits = None
    constrained_choice = None
    correct_constrained = None
    if prepositions is not None and question_format != "free":
        option_logits = {}
        for prep in sorted(prepositions):
            prep_ids = processor.tokenizer.encode(prep, add_special_tokens=False)
            prep_first_id = prep_ids[0]
            logit = first_token_scores[0, prep_first_id].item()
            option_logits[prep] = float(logit)
        constrained_choice = max(option_logits, key=option_logits.get)
        correct_constrained = constrained_choice == ground_truth

    # Free format: regex match on full generated text
    correct_regex = None
    if question_format == "free":
        correct_regex = bool(
            re.search(rf'\b{re.escape(ground_truth)}\b', generated_text, re.IGNORECASE)
        )

    result = {
        "question": question,
        "question_format": question_format,
        "objects": objects,
        "ground_truth": ground_truth,
        "generated_text": generated_text,
        "first_generated_token": first_generated_token,
        "first_generated_id": first_generated_id,
        "gt_first_token_id": gt_first_token_id,
        "gt_token_ids": gt_token_ids,
        "argmax_token": argmax_token,
        "generate_argmax_match": generate_argmax_match,
        "max_logit": float(max_logit),
        "top5_tokens": top5_tokens,
        "top5_ids": top5_ids_list,
        "top5_logits": top5_logits,
        # Accuracy methods
        "correct_first_token_id": correct_first_token_id,
        "correct_top5_token_id": correct_top5_token_id,
    }
    if correct_regex is not None:
        result["correct_regex"] = correct_regex
    if option_logits is not None:
        result["option_logits"] = option_logits
        result["constrained_choice"] = constrained_choice
        result["correct_constrained"] = correct_constrained

    return result


def run_baseline(
    dataset_path, dataset_name, model_path, n_samples=None, no_image=False,
    question_format="forced_choice", task_type="spatial_relative",
    perspective=None,
):
    """Run baseline evaluation on dataset."""
    # Load model and dataset using shared utilities
    model, processor = load_model(model_path)
    dataset, prepositions = load_dataset(dataset_path, dataset_name, task_type=task_type,
                                         perspective=perspective)

    # Determine sample count
    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))

    mode = "NO IMAGE" if no_image else "with image"
    print(f"\nEvaluating {n_samples} samples ({mode}, format={question_format})...")

    results = []
    for idx in tqdm(range(n_samples), desc="Evaluating"):
        sample = dataset[idx]
        image = sample["image"]
        objects = sample["objects"]
        ground_truth = sample["preposition"]

        question = create_question(objects, prepositions, question_format=question_format,
                                   task_type=task_type, perspective=perspective)

        # Print the prompt for the first sample
        if idx == 0:
            messages = (
                [{"role": "user", "content": [{"type": "text", "text": question}]}]
                if no_image
                else [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            )
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print(f"\n--- Prompt for sample 0 ---\n{prompt_text}\n---\n")

        result = evaluate_sample_baseline(
            model, processor, image, question, ground_truth, objects,
            no_image=no_image, question_format=question_format,
            prepositions=prepositions,
        )
        result["sample_idx"] = idx
        results.append(result)

        torch.cuda.empty_cache()

    # Compute summary stats
    n_evaluated = len(results)

    # Check generate vs argmax consistency
    n_match = sum(r["generate_argmax_match"] for r in results)

    # Accuracy (using knockout method: first token ID match)
    acc_top1 = sum(r["correct_first_token_id"] for r in results) / n_evaluated * 100
    acc_top5 = sum(r["correct_top5_token_id"] for r in results) / n_evaluated * 100

    summary = {
        "dataset_name": dataset_name,
        "question_format": question_format,
        "no_image": no_image,
        "n_samples": n_evaluated,
        "generate_argmax_match_rate": n_match / n_evaluated * 100,
        "accuracy_top1": acc_top1,
        "accuracy_top5": acc_top5,
    }

    # Regex accuracy for free format
    if question_format == "free":
        acc_regex = sum(r.get("correct_regex", False) for r in results) / n_evaluated * 100
        summary["accuracy_regex"] = acc_regex

    # Constrained accuracy: argmax over valid option logits
    has_constrained = any(r.get("correct_constrained") is not None for r in results)
    if has_constrained:
        acc_constrained = sum(r.get("correct_constrained", False) for r in results) / n_evaluated * 100
        summary["accuracy_constrained"] = acc_constrained

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Question format: {question_format}")
    print(f"Samples evaluated: {n_evaluated}")
    print()
    print(
        f"Generate vs Argmax match rate: {summary['generate_argmax_match_rate']:.1f}%"
    )
    print()
    if question_format == "open" and has_constrained:
        print("ACCURACY (constrained — argmax over valid options):")
        print(f"  Accuracy: {acc_constrained:.1f}%")
        print()
        print("  (also stored: top-1 full vocab = {:.1f}%, top-5 = {:.1f}%)".format(acc_top1, acc_top5))
    else:
        print("ACCURACY:")
        print(f"  Top-1 (first token): {acc_top1:.1f}%")
        print(f"  Top-5 (first token): {acc_top5:.1f}%")
        if has_constrained:
            print(f"  Constrained (argmax over options): {acc_constrained:.1f}%")
    if question_format == "free":
        print(f"  Regex (word match):  {summary['accuracy_regex']:.1f}%")
    print("=" * 70)

    # Show some mismatches if any
    mismatches = [r for r in results if not r["generate_argmax_match"]]
    if mismatches:
        print(f"\nGenerate/Argmax mismatches ({len(mismatches)}):")
        for r in mismatches[:5]:
            print(
                f"  Sample {r['sample_idx']}: generate='{r['first_generated_token']}', argmax='{r['argmax_token']}'"
            )

    # Show some incorrect samples
    if question_format == "free":
        incorrect = [r for r in results if not r.get("correct_regex", False)]
    else:
        incorrect = [r for r in results if not r["correct_first_token_id"]]
    if incorrect:
        print(f"\nIncorrect samples ({len(incorrect)}):")
        for r in incorrect[:10]:
            print(f"  Sample {r['sample_idx']}:")
            print(f"    Question: {r['question']}")
            print(f"    Objects: {r['objects']}")
            if question_format == "free":
                print(
                    f"    GT: '{r['ground_truth']}', Generated: '{r['generated_text']}'"
                )
            else:
                print(
                    f"    GT: '{r['ground_truth']}', Got: '{r['first_generated_token']}', Top5: {r['top5_tokens']}"
                )

    return results, summary


def save_results(results, summary, output_dir, dataset_name=None):
    """Save results to JSON."""
    if dataset_name:
        output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-sample results
    with open(output_dir / "baseline_samples.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Per-sample results saved to {output_dir / 'baseline_samples.json'}")

    # Save summary
    with open(output_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {output_dir / 'baseline_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Baseline forward pass")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--dataset-path", type=str, help="Base path to datasets")
    parser.add_argument("--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument("--model-path", type=str, help="Path to Qwen3-VL model")
    parser.add_argument("--n-samples", type=int, help="Number of samples to process")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument(
        "--no-image", action="store_true", help="Run without image (text-only baseline)"
    )
    parser.add_argument(
        "--question-format",
        type=str,
        default="forced_choice",
        choices=["forced_choice", "open", "free"],
        help="Question format: forced_choice (with alternatives), "
             "open (single word, no alternatives), free (bare question, 20 tokens).",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="spatial_relative",
        choices=["spatial_relative", "spatial_absolute", "recognition", "attribute_chain", "attribute_shape"],
        help="Task type: spatial_relative (2 objects, default), "
             "spatial_absolute (1 object, absolute position), "
             "recognition (what object is in image), "
             "attribute_chain (color of object at direction from reference), "
             "attribute_shape (shape of object at direction from reference).",
    )
    parser.add_argument(
        "--perspective",
        type=str,
        default=None,
        choices=["camera", "addressee", "main"],
        help="Perspective prefix for POV questions (COMFORT dataset). "
             "camera: 'From the camera's viewpoint, ...', "
             "addressee: 'From the woman's viewpoint, ...', "
             "main: 'From the car's viewpoint, ...'.",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=None,
        help="Override results subdirectory (e.g., 'baseline_8b').",
    )

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)

    # Override with CLI args
    if args.dataset_path is not None:
        config["dataset_path"] = args.dataset_path
    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    if args.model_path is not None:
        config["model_path"] = args.model_path
    if args.n_samples is not None:
        config["n_samples"] = args.n_samples
    if args.no_save:
        config["save"] = False

    # Set defaults
    config.setdefault("save", True)
    config.setdefault("n_samples", None)

    # Validate
    required = ["dataset_path", "dataset_name", "model_path"]
    missing = [k for k in required if k not in config]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    # Run
    results, summary = run_baseline(
        dataset_path=config["dataset_path"],
        dataset_name=config["dataset_name"],
        model_path=config["model_path"],
        n_samples=config["n_samples"],
        no_image=args.no_image,
        question_format=args.question_format,
        task_type=args.task_type,
        perspective=args.perspective,
    )

    # Save
    if config["save"]:
        if args.results_subdir:
            subdir = args.results_subdir
        elif args.no_image:
            subdir = "no_image"
        elif args.question_format != "forced_choice":
            subdir = f"baseline_{args.question_format}"
        elif args.task_type != "spatial_relative":
            subdir = f"baseline_{args.task_type}"
        else:
            subdir = "baseline"
        ds_name = config["dataset_name"]
        if args.perspective:
            ds_name = f"{ds_name}_{args.perspective}"
        output_dir = Path(__file__).parent.parent / "results" / subdir
        save_results(results, summary, output_dir, dataset_name=ds_name)


if __name__ == "__main__":
    main()
