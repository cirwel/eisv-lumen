---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-4B
tags:
  - eisv
  - dynamics
  - trajectory
  - expression-generation
  - lora
datasets:
  - hikewa/unitares-eisv-trajectories
pipeline_tag: text-generation
---

# EISV-Lumen Teacher — LoRA on Qwen3-4B

A LoRA fine-tune of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) that maps **EISV trajectory shapes** to **primitive expression tokens**. Given a sequence of EISV dimension values and a classified trajectory shape, the model generates a structured expression — a short sequence of emotion-like tokens that a downstream embodied system (Lumen, running on a Raspberry Pi 4) uses to drive LED color, screen drawing, and inner-voice narration.

This is the **teacher model** in a teacher-student pipeline. Its outputs are distilled into a lightweight RandomForest student that runs on-device.

## Model Details

| Detail | Value |
|--------|-------|
| **Base model** | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| **Method** | LoRA (Low-Rank Adaptation) |
| **Rank (r)** | 16 |
| **Alpha** | 32 |
| **Dropout** | 0.1 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Task type** | CAUSAL_LM |
| **Training examples** | 2,880 (50% real + 50% synthetic) |
| **Training steps** | 1,260 |
| **Adapter size** | 126 MB |
| **PEFT version** | 0.18.1 |
| **License** | Apache 2.0 |

## What is EISV?

EISV is a four-coordinate state estimate. In the real-data portion of this model's training pipeline, the coordinates are derived from Lumen's physical sensors and computational telemetry. The training set also includes synthetic trajectories; sensor provenance does not apply to those examples.

The [companion dataset card](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories/blob/6e9ab8aa3e20290c47477a15b23021b7ae7c8421/README.md) documents the Lumen observation-layer mapping:

| Dimension | Name | Computed coordinate |
|-----------|------|---------------------|
| **E** | Energy | Sensor-derived `warmth`; not measured productive capacity |
| **I** | Information Integrity | Sensor-derived `clarity`; not validated reasoning quality or integrity |
| **S** | Entropy | `1 − stability`; not semantic uncertainty or Shannon entropy |
| **V** | Void | `(1 − presence) × 0.3`; an observation-layer proxy, not the signed governance-layer imbalance integrator |

The coordinate names and expression tokens are modeling conventions, not measurements of subjective experience. Expression coherence below is a token-to-shape fit score, distinct from UNITARES governance coherence; it does not validate safety or reasoning quality.

### Training provenance and sampling cadence

The v6 training configuration describes a blend of 1,803 real-derived and 1,797 synthetic examples before splitting. The retained local files contain 2,880 training, 360 validation, and 360 test examples. The linked Hub dataset is a larger trajectory-window corpus, not the 2,880-example training split itself.

The pinned companion dataset card reports typical sensor-driven intervals of approximately 2 seconds, with actual timestamps in each window. The earlier 30-second claim in this model card was not tied to a documented sampling or downsampling stage and has been removed. This dataset-card revision documents the coordinate mapping and corpus; it is **not** a verified pin of the teacher's original training input. The retained formatted training files contain text, shape, pattern, and messages but no source timestamps or Hub revision, so they do not establish the original training-window cadence or exact upstream dataset cut.

## Trajectory Shapes

A trajectory is a time-window of EISV values. The shape classifier identifies 9 patterns. The frequencies below are approximate historical corpus summaries, not counts for the current Hub dataset or the balanced training split. Shape names describe coordinate patterns, not independently measured psychological states:

| Shape | Description | Real-data frequency |
|-------|-------------|-------------------|
| `settled_presence` | Small coordinate changes with low S and V | 47% |
| `convergence` | Dimensions moving toward alignment | 41% |
| `entropy_spike_recovery` | Sharp entropy increase followed by return to baseline | 5% |
| `basin_transition_up` | Shift from lower to higher energy basin | 2% |
| `basin_transition_down` | Shift from higher to lower energy basin | 2% |
| `rising_entropy` | Sustained increase in the S coordinate | 1.5% |
| `falling_energy` | Gradual decline in the E coordinate | 1.5% |
| `void_rising` | Increasing V coordinate | 0.3% |
| `drift_dissonance` | Conflicting dimensional movements | synthetic only |

The distribution is heavily skewed toward `settled_presence` and `convergence` in the historical corpus; this distribution alone does not establish device health or agent capability. The training set uses 50/50 real+synthetic blending to ensure the model sees enough rare shapes.

## Expression Vocabulary

The model generates structured expressions using 15 primitive tokens and 5 patterns.

### Tokens (15)

`~warmth~` `~curiosity~` `~resonance~` `~stillness~` `~boundary~` `~reaching~` `~reflection~` `~ripple~` `~deep_listening~` `~emergence~` `~questioning~` `~holding~` `~releasing~` `~threshold~` `~return~`

### Patterns (5)

| Pattern | Structure | Example |
|---------|-----------|---------|
| SINGLE | One token | `~stillness~` |
| PAIR | Two tokens | `~warmth~ ~resonance~` |
| TRIPLE | Three tokens | `~curiosity~ ~reaching~ ~emergence~` |
| REPETITION | Repeated token (intensity) | `~warmth~ ~warmth~ ~warmth~` |
| QUESTION | Token with `?` (uncertainty) | `~threshold~?` |

## Evaluation Results

Evaluated on **500 real trajectories** (no synthetic data in the test set):

| Metric | Score |
|--------|-------|
| **Mean coherence** | 0.952 |
| **Valid rate** | 100% |
| **Pattern accuracy** | 25.8% |

**Coherence** measures how well the generated expression fits the trajectory shape (semantic alignment between tokens and shape meaning). **Valid rate** measures whether outputs parse as legal expressions. **Pattern accuracy** measures exact match of the structural pattern — the relatively low score is expected since multiple patterns can be coherent for a given shape.

### Per-Shape Coherence

| Shape | Coherence |
|-------|-----------|
| settled_presence | 0.993 |
| convergence | 0.936 |
| void_rising | 1.000 |
| basin_transition_down | 1.000 |
| basin_transition_up | 1.000 |
| rising_entropy | 1.000 |
| falling_energy | 0.875 |
| entropy_spike_recovery | 0.833 |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Apply LoRA adapter
model = PeftModel.from_pretrained(base_model, "hikewa/eisv-lumen-teacher")

# Format input prompt
prompt = """You are an EISV expression generator. Given trajectory dynamics, produce a primitive expression.

Trajectory shape: settled_presence
EISV values: E=0.72, I=0.85, S=0.15, V=0.08
Dominant dimension: I (Information Integrity)
Trend: stable

Expression:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    temperature=0.7,
    do_sample=True,
)
result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)
# Example output: ~stillness~ ~resonance~
```

## Related

- **Dataset**: [hikewa/unitares-eisv-trajectories](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories) — companion trajectory-window corpus; see the training-provenance limits above
- **Student model**: [hikewa/eisv-lumen-student](https://huggingface.co/hikewa/eisv-lumen-student) — distilled RandomForest that runs on Raspberry Pi 4
- **Demo**: [hikewa/eisv-lumen-explorer](https://huggingface.co/spaces/hikewa/eisv-lumen-explorer) — interactive Space to explore trajectories and expressions

## Citation

```bibtex
@misc{eisv-lumen-teacher-2026,
  title={EISV-Lumen Teacher: LoRA Fine-Tune for Trajectory-to-Expression Generation},
  author={hikewa},
  year={2026},
  url={https://huggingface.co/hikewa/eisv-lumen-teacher},
  note={LoRA adapter on Qwen/Qwen3-4B mapping EISV trajectory shapes to primitive expression tokens}
}
```
