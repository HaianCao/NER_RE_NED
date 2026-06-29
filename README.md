# Clinical NER + RE Fine-tuning Pipeline

Instruction-tuned LLM pipeline for **Named Entity Recognition (NER)** and
**Relation Extraction (RE)** on clinical text in **English and Vietnamese**,
built on top of [Unsloth](https://github.com/unslothai/unsloth) + LoRA.

---

## Project Structure

```
NER_RE_NED/
├── config.py                 # PipelineConfig, enums (TaskMode, TrainingFormat, REMode)
├── data/
│   ├── schemas.py            # EntitySpan, RelationTriple, StandardizedDocument
│   ├── adapter.py            # ClinicalFormatAdapter (EN/VI auto-detect)
│   └── registry.py           # DatasetRegistry (multi-file aggregation)
├── prompts/
│   ├── builder.py            # BasePromptBuilder (ABC)
│   ├── causal.py             # Format A: plain text input → JSON
│   ├── instruction.py        # Format B: System / User / Assistant (ChatML)
│   ├── factory.py            # PromptBuilderFactory
│   └── _helpers.py           # Shared text formatters (no side-effects)
├── training/
│   ├── collator.py           # DataCollatorWithLossMask (prompt tokens masked)
│   ├── metrics.py            # NER / RE F1 for post-training evaluation
│   ├── model_factory.py      # UnslothModelFactory (LoRA setup)
│   └── trainer.py            # NativeSafeTrainer (multi-GPU safe)
├── utils/
│   └── span_utils.py         # resolve_span_to_index, normalize_term
├── configs/
│   └── default.yaml          # ← Edit this to configure your experiment
├── train.py                  # Entry point for accelerate launch
├── requirements.txt
└── README.md
```

---

## Configuration

All experiment settings live in **`configs/default.yaml`**.
No code changes needed between experiments — just edit the YAML.

| Key | Options | Default |
|---|---|---|
| `training.format` | `causal` \| `instruction` | `instruction` |
| `training.task_mode` | `ner_only` \| `re_only` \| `joint` | `joint` |
| `training.re_mode` | `pairwise` \| `global` | `global` |
| `model.name` | Any Unsloth model slug | `Phi-3-mini-4k-instruct-bnb-4bit` |

### Suggested Experiments

| # | task_mode | re_mode | format | language |
|---|---|---|---|---|
| 1 (baseline) | `re_only` | `pairwise` | `instruction` | EN only |
| 2 | `re_only` | `global` | `instruction` | EN only |
| 3 (end-to-end) | `joint` | `global` | `instruction` | EN + VI |
| 4 (causal) | `joint` | `global` | `causal` | EN + VI |

---

## Running on Kaggle Notebook

### Prerequisites
- Kaggle notebook with **2× GPU** (e.g. 2× T4 or P100)
- Dataset `hiancao/n2c2-eng` added as a data source

### Cell 1 — Install Unsloth and dependencies
```python
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
!pip install pyyaml
```

### Cell 2 — Clone this repository
```python
!git clone https://github.com/YOUR_USERNAME/NER_RE_NED.git /kaggle/working/NER_RE_NED
```
> Replace `YOUR_USERNAME` with your GitHub username.

### Cell 3 — (Optional) Modify the config inline
```python
import yaml

CONFIG_PATH = "/kaggle/working/NER_RE_NED/configs/default.yaml"
MY_CONFIG   = "/kaggle/working/my_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# ── Change experiment settings here ──────────────────────────
cfg["training"]["task_mode"]  = "joint"        # ner_only | re_only | joint
cfg["training"]["format"]     = "instruction"  # causal   | instruction
cfg["training"]["re_mode"]    = "global"       # pairwise | global
cfg["model"]["name"]          = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
# ─────────────────────────────────────────────────────────────

# Add / remove training files
cfg["data"]["train_files"] = [
    {"path": "/kaggle/input/n2c2-eng/n2c2_2010_train_en.jsonl", "language": "auto"},
    {"path": "/kaggle/input/n2c2-eng/n2c2_2010_train_vi.jsonl", "language": "auto"},
]
# Dev files for Early Stopping
cfg["data"]["eval_files"] = [
    {"path": "/kaggle/input/n2c2-eng/n2c2_2010_test_en.jsonl", "language": "auto"},
    {"path": "/kaggle/input/n2c2-eng/n2c2_2010_test_vi.jsonl", "language": "auto"},
]
# Test files for Final Evaluation (F1/Precision/Recall)
cfg["data"]["test_files"] = [
    {"path": "/kaggle/input/n2c2-eng/n2c2_2010_test_en.jsonl", "language": "auto"},
    {"path": "/kaggle/input/n2c2-eng/n2c2_2010_test_vi.jsonl", "language": "auto"},
]

# Fast validation / Early stopping tweaks
cfg["training"]["eval_subset_size"] = 0.2     # Use 20% of Dev set during training for speed
cfg["training"]["early_stopping_patience"] = 3 # Stop if loss doesn't improve for 3 checks


with open(MY_CONFIG, "w") as f:
    yaml.dump(cfg, f, allow_unicode=True)

print("Config saved to:", MY_CONFIG)
```

### Cell 4 — Configure accelerate for 2 GPUs
```python
!accelerate config default
```

### Cell 5 — Launch training on 2 GPUs
```python
CONFIG = "/kaggle/working/my_config.yaml"   # or use default.yaml

!accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    /kaggle/working/NER_RE_NED/train.py \
    --config {CONFIG} \
    2>&1 | tee /kaggle/working/training_log.txt
```

> **Note:** The `2>&1 | tee` part captures both stdout and stderr into
> `training_log.txt` while still showing output in the notebook.

### Cell 6 — Evaluate the trained model (Test Phase)

During training, the model only tracks `eval_loss` on the Dev set (fast).
To calculate actual F1, Precision, and Recall, we must generate text on the **Test set** (which is slower).

```python
CONFIG = "/kaggle/working/my_config.yaml"   # or use default.yaml

!accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    /kaggle/working/NER_RE_NED/evaluate.py \
    --config {CONFIG} \
    --checkpoint /kaggle/working/outputs/final_lora_adapter \
    --batch_size 16 \
    --output /kaggle/working/predictions.jsonl
```

### Cell 7 — (Optional) Explore the output directory
```python
import os
print(os.listdir("/kaggle/working/outputs/final_lora_adapter"))
print(os.listdir("/kaggle/working/outputs/final_merged_model"))
```

---

## Label Ontology (n2c2 2010)

### Entity Labels
| Label | Description |
|---|---|
| `Problem` | Diseases, symptoms, disorders, injuries |
| `Treatment` | Medications, surgeries, therapies, devices |
| `Test` | Lab tests, imaging studies, clinical measurements |

### Relation Labels
| Label | Subject → Object | Description |
|---|---|---|
| `TrIP` | Treatment → Problem | Treatment **Improves** Problem |
| `TrWP` | Treatment → Problem | Treatment **Worsens** Problem |
| `TrCP` | Treatment → Problem | Treatment **Causes** Problem |
| `TrAP` | Treatment → Problem | Treatment **Administered for** Problem |
| `TrNAP` | Treatment → Problem | Treatment **Not Administered because of** Problem |
| `TeRP` | Test → Problem | Test **Reveals** Problem |
| `TeCP` | Test → Problem | Test **Conducted because of** Problem |
| `PIP` | Problem → Problem | Problem **Indicates** Problem |

---

## SOLID Design Notes

| Principle | Implementation |
|---|---|
| **S**ingle Responsibility | `ClinicalFormatAdapter` parses only; `PromptBuilder` formats only |
| **O**pen/Closed | New formats added via new `BasePromptBuilder` subclasses |
| **L**iskov Substitution | `CausalPromptBuilder` and `InstructionPromptBuilder` are fully interchangeable |
| **I**nterface Segregation | `BasePromptBuilder` has one abstract method |
| **D**ependency Inversion | `train.py` depends on `BasePromptBuilder`, not concrete classes |
