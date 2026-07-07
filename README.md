# Pathways of Visual Information Flow in Vision-Language Models

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Models

Set `model_path` in the config to a Hugging Face id (downloaded on first run) or a local directory.

| Config | `model_path` |
|--------|--------------|
| `configs/default.yaml` | `Qwen/Qwen3-VL-4B-Instruct` |
| `configs/llava_15_7b.yaml` | `llava-hf/llava-1.5-7b-hf` |
| `configs/internvl3_5_4b.yaml` | `OpenGVLab/InternVL3_5-4B-HF` |

## Data

Build scripts live in `data/`; every experiment reads `<dataset_path>/<dataset-name>.hf` and `dataset_path` defaults to `./data`.

- Paired experiments (causal patching, input dominance) need a `pair_id`. Synthetic sets are paired; for natural relations/localization use the `*_hflip` variants (`coco_two_hflip`, `vg_qa_two_hflip`, `coco_one_hflip`, `vg_qa_one_hflip`, …).
- `--question-format`: `open` (single word) or `forced_choice` (options in prompt).
- Change the model by swapping `--config`. Override paths with `--model-path` / `--dataset-path`.

### Building the datasets

All outputs are written to `./data`.

**Synthetic** (generated, no downloads):
```bash
python data/generate_shapes_dataset.py --n-samples 200 --center --pairs --aligned --output data/controlled_shapes_pairs.hf
python data/generate_shapes_dataset.py --n-samples 200 --single position    --output data/controlled_shapes_localization.hf
python data/generate_shapes_dataset.py --n-samples 200 --single recognition  --output data/controlled_shapes_recognition.hf
```

**Natural**. Download the source benchmarks, then run the transform:
*What's-Up* (→ `controlled_images`, `controlled_clevr`, `coco_one`, `coco_two`, `vg_qa_one`, `vg_qa_two`).
Downloads automatically from Google Drive (Kamath et al., 2023):
```bash
python data/build_whatsup.py --data-dir ./data
```

*VSR* (→ `vsr_spatial`). Download the VSR dataset (Liu et al., 2023) in HF on-disk format to `data/vsr_raw.hf`, then:
```bash
python data/convert_vsr.py --input ./data/vsr_raw.hf --output ./data/vsr_spatial.hf
```

*COCO recognition* (→ `coco_recognition`, `coco_recognition_pairs`). Needs COCO `train2017` images + annotations and the question set from llava-interp (Neo et al., 2024):
```bash
wget https://raw.githubusercontent.com/clemneo/llava-interp/main/data/clean_questions.json -O data/clean_questions.json
python data/create_recognition_dataset.py \
    --coco-images /path/to/train2017 \
    --coco-annotations /path/to/annotations/instances_train2017.json \
    --questions data/clean_questions.json \
    --output data/coco_recognition.hf
python data/create_recognition_pairs.py --dataset-path data --dataset-name coco_recognition
```

**Horizontal-flip pairs** (→ `*_hflip`, needed for paired natural experiments), built from the
What's-Up outputs above:
```bash
python data/create_mirror_pairs.py --dataset-path data --datasets coco_two vg_qa_two coco_one vg_qa_one
```

**Object-swapped control** (→ `controlled_shapes_pairs_swapped`):
```bash
python data/create_swapped_pairs.py --dataset-path data --dataset-name controlled_shapes_pairs
```

## Experiments
These experiments assume that you have created the datasets.

### Baseline
```bash
python experiments/baseline.py --config configs/default.yaml \
    --dataset-name controlled_shapes_pairs --task-type spatial_relative --question-format open
# add --question-format forced_choice for choices, --no-image for text-only
```

COCO recognition (open-vocabulary answers) uses dedicated scripts:
```bash
python experiments/baseline_recognition_pairs.py --config configs/default.yaml --dataset-name coco_recognition_pairs  # open
python experiments/baseline_recognition_fc.py    --config configs/default.yaml --dataset-name coco_recognition_pairs  # choices
```

### Attention knockout
```bash
python experiments/knockout.py --config configs/default.yaml \
    --dataset-name controlled_shapes_pairs --task-type spatial_relative --question-format open
```

### Causal patching (paired)
```bash
python experiments/causal_tracing.py --config configs/default.yaml \
    --dataset-name controlled_shapes_pairs --task-type spatial_relative \
    --groups image,all_text,last_token --n-pairs 200 --question-format open
# natural data: use the *_hflip dataset
```

### Corrupted patching (unpaired)
```bash
python experiments/causal_tracing.py --config configs/default.yaml \
    --dataset-name coco_two --task-type spatial_relative \
    --corruption noise --n-pairs 200 --question-format open
```

### Source dominance
```bash
python experiments/text_dominance.py --config configs/default.yaml \
    --dataset-name coco_two_hflip --task-type spatial_relative --n-pairs 200
# COCO recognition: --task-type recognition --use-question-text --prefill "It is a"
```

### Text recovery
```bash
python experiments/text_recovery.py --config configs/default.yaml \
    --dataset-name coco_two --task-type spatial_relative \
    --corruption noise --question-format open --n-samples 200   # --corruption: noise (default) | black | white
```

### Recognition (open-vocabulary)
```bash
# paired causal patching
python experiments/causal_tracing_generate.py --config configs/default.yaml \
    --dataset-name coco_recognition_pairs --task-type recognition \
    --prefill "It is a" --use-question-text --n-pairs 100
# paired causal patching, 4-option forced choice
python experiments/causal_tracing_recognition_fc.py --config configs/default.yaml \
    --dataset-name coco_recognition_pairs --n-pairs 200
# corrupted patching
python experiments/corrupted_tracing_generate.py --config configs/default.yaml \
    --dataset-name coco_recognition --corruption black --n-samples 100
```