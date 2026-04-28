# LLM-distiller
A demonstration of **offline SFT (Supervised Fine-Tuning) distillation** where a large teacher LLM transfers knowledge of a made-up programming language to a tiny student model.

## Contributors

<a href="https://github.com/LeOvO7/LLM-distiller/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=LeOvO7/LLM-distiller" />
</a>

## Overview

| Component | Detail |
|---|---|
| Fictional Language | **Floq** — invented syntax with unique rules |
| Teacher Model | OpenAI `gpt-4o-mini` |
| Student Model | `Qwen/Qwen2.5-0.5B-Instruct` + LoRA |
| Fine-tuning Method | LoRA via `trl` / `peft` |
| Environment | Google Colab (NVIDIA A100-SXM4-40GB) |

## Results

| Stage | Floq Keyword Hit Rate |
|---|---|
| Baseline (no training) | **5.2%** |
| After Distillation | **31.2%** |
| Improvement | **+505.9%** |

An additional **Reverse Distillation** experiment showed that injecting the student's outputs as few-shot examples into GPT-4o-mini raised its Floq hit rate from 1.8% → 27.3%, demonstrating bidirectional knowledge transfer.

---

## The Floq Language

Floq is a fictional language designed to have no overlap with any real language the student model has seen during pre-training. This makes it an ideal controlled domain for measuring knowledge transfer.

```floq
## Compute factorial
fn factorial(n) =>
  when n <= 1 { 1 } else { n * factorial(n - 1) }

val result := factorial(5)
shout(result)  ## prints 120
```

### Syntax Rules

| Feature | Syntax |
|---|---|
| Variable declaration | `val x := 42` |
| Function definition | `fn add(a, b) => a + b` |
| Conditional | `when x > 0 { ... } else { ... }` |
| Loop | `loop 10 times { ... }` |
| Print / Output | `shout("hello")` |
| Types | `num`, `str`, `bool`, `list` |
| Comments | `## This is a comment` |
| List literal | `val xs := [1, 2, 3]` |
| List indexing | `xs~0` (tilde `~` replaces `[]`) |
| String concatenation | `"hello" ++ " world"` |
| Boolean operators | `and`, `or`, `not` |

---

## Pipeline

```
Step 0  Install dependencies
Step 1  Load API key & verify GPU
Step 2  Define Floq spec + generate Q&A dataset (teacher: GPT-4o-mini)
Step 3  Evaluate baseline student (Qwen2.5-0.5B, no training)
Step 4  LoRA fine-tuning on 200 teacher-generated samples
Step 5  Evaluate distilled student on 30-item test set
Step 6  Visualize results (bar chart, per-keyword breakdown, score distribution)
Step 7* Reverse Distillation — student teaches the LLM via few-shot injection
```

---

## Dataset Generation

The teacher model (`gpt-4o-mini`) is prompted with the full Floq spec and asked to answer randomized question templates. This produces labeled `(question, answer)` pairs:

- **Training set**: 200 items
- **Test set**: 30 items
- Saved as `floq_data/train.json` and `floq_data/test.json`

Question categories include syntax queries, code comprehension, invalid-code detection, and full program writing tasks.

---

## LoRA Configuration

```python
LoraConfig(
    r=32,                  # rank
    lora_alpha=64,         # scaling factor
    target_modules=[       # layers injected with LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
```

- **Trainable parameters**: 17.6 M out of 511.6 M (3.44%)
- **Training time**: ~56.5 seconds on A100
- **Final training loss**: 0.4454

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Effective batch size | 16 (4 × 4 accumulation steps) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Optimizer | `paged_adamw_32bit` |
| Precision | FP16 |
| Max sequence length | 512 |

---

## Evaluation Metric

**Floq Keyword Hit Rate** — measures what fraction of 11 Floq-specific syntax tokens appear in the model's response:

```
:=   fn    =>   when   shout(   loop   times   ~   val   ++   ##
```

A score of 1.0 means the model used all 11 keywords in its answer. This metric directly measures whether the model has internalized Floq syntax rather than falling back to Python/JavaScript conventions.

---

## Reverse Distillation (Step 7)

After distillation, the student model generates 5 high-quality Floq Q&A examples. These are injected as few-shot context into the GPT-4o-mini system prompt, creating a loop where the small model teaches the large one:

```
Teacher → (distillation) → Student → (few-shot injection) → Teacher (Floq-aware)
```

| Condition | Score |
|---|---|
| GPT-4o-mini (no Floq knowledge) | 1.8% |
| GPT-4o-mini (student few-shot injection) | 27.3% |

---

## Dependencies

```
transformers==4.44.0
peft==0.12.0
trl==0.10.1
datasets
accelerate
bitsandbytes
openai
sentencepiece
protobuf
rouge-score
```

## Output Files

| File | Description |
|---|---|
| `floq_data/train.json` | 200 teacher-generated training samples |
| `floq_data/test.json` | 30 test samples |
| `floq_student_lora/` | Saved LoRA adapter weights |
| `distillation_results.png` | Visualization charts |
| `floq_distillation_results.zip` | All outputs packaged for download |


[View Detailed Output Data](https://drive.google.com/file/d/1opjgyoRSMG-_J3jDcrAzZ6eVTJ143Shm/view?usp=sharing)
