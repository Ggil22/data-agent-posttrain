# data-agent-posttrain

Post-training a small open model (SFT, then DPO) to work as a data analyst agent, and
measuring what each stage buys on [data-agent-bench](https://github.com/Ggil22/data-agent-bench).

> **Status:** phase 4 of 6 (DPO). SFT is trained and evaluated; DPO is implemented, not run yet.

## Design

**No benchmark leakage.** data-agent-bench's 65 tasks are never trained on. Training uses
new synthetic domains with their own tables, traps and question templates; a test checks that
no training table matches a benchmark table and that no training question is close to a
benchmark question.

**Verifiable rewards.** Every task's answer is computed by code from the data, so a
trajectory is scored without an LLM judge. That gives rejection-sampled SFT data and
on-policy DPO pairs (a correct and a failed trajectory for the same task) almost for free.

**Three evaluation levels.**

| Split | What it measures | Tasks |
|---|---|---|
| `val` | Same domains, unseen data (new seeds) | 300 |
| `heldout` | A domain never trained on (`inventory`) | 180 |
| data-agent-bench | Different tables, traps and question styles | 65 |

## Training domains

Each domain is a generator of several related CSV tables plus a `DATA_DICTIONARY.md` that
documents the traps, as in real projects. Each seed gives new data and new question
parameters.

| Domain | Tables | Traps |
|---|---|---|
| `rentals` | stations, trips | Duplicate rows, staff trips to exclude, `-1` unknown durations, hand-typed rider types, empty fare = 0 |
| `clinic` | doctors, patients, appointments | Two date formats (incl. `DD.MM.YYYY`), app fees in cents, test patients, a no-show penalty rule |
| `energy` | meters, readings | Wh vs kWh meters, a `99999.9` error code, missing days, a weekday/weekend tariff |
| `inventory` (held out) | products, movements | `YYYYMMDD` dates, SKU case variants, a TEST warehouse, signed stock from in/out movements |

Templates span ten categories (aggregation, groupby, filtering, time series, join, cleaning,
statistics, multi-step, business rules and unanswerable), and about 18% of the tasks are
unanswerable. `dapt build` writes 2,979 training, 300 validation and 180 held-out tasks.

A first check with `gemini-3.5-flash-lite` on 17 validation tasks solved 8. In all 9 failures
the code was reasonable but the model never opened the data dictionary, so duplicates,
excluded rows and sentinel values went into the answer. Reading the documentation before
analysing is exactly the habit the training should teach.

## Plan

1. **Training tasks** (done): domains, templates, splits, leakage tests.
2. **Baseline** (done): `Qwen3-4B-Instruct-2507` served with vLLM on Kaggle (2x T4).
3. **SFT** (done): guided rejection sampling from the base model, LoRA SFT on 1,219 examples.
4. **DPO** (implemented): the SFT model sampled 4 times per task it was not trained on; a
   correct and a failed episode of the same task make a pair (for never-solved tasks the
   correct side comes from the SFT sampling). The SFT adapter is trained further with DPO
   plus an NLL term on the chosen side; reference log-probs are computed before training.

| Exact match | val | heldout | data-agent-bench |
|---|---|---|---|
| Base, strict | 4/300 (1%) | 6/180 (3%) | 2/65 (3%) |
| Base, format repair | 87/300 (29%) | 78/180 (43%) | 46/65 (71%) |
| SFT, strict | 182/300 (61%) | 78/180 (43%) | 52/65 (80%) |

SFT removes the format errors and learns the trained domains' traps, but does not transfer to the
held-out domain (aggregation, cleaning, business rules and multi-step: 0/48 on `inventory`).
42 of its 480 episodes ended in a 120 s API timeout, counted as failures.
5. **Report:** base vs SFT vs SFT+DPO vs API models, with confidence intervals, the failure
   taxonomy and a general-capability check for forgetting.
6. **Release:** LoRA adapters and the dataset on the Hugging Face Hub.

## Usage

```bash
uv sync
uv run dapt build                       # task sets in data/tasks/
uv run dapt rollout --model gemini-flash-lite --split val --limit 20
uv run pytest
```

Kaggle jobs, in order: `kaggle/generate` (SFT samples), `kaggle/train` (SFT),
`kaggle/dpo-generate` (on-policy samples), `kaggle/dpo-train` (DPO), `kaggle/evaluate`.
