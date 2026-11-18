# CLAUDE.md

This file guides Claude Code when it works in this repository.

## What this repo is

This is a student's working copy of the CSCI 5942 (AI Engineering, Fall 2026, CU Boulder) assignments repo. It has two remotes:

- `upstream` = `csci5942/assignments`. The instructors release each assignment here and sometimes correct it in place. Pull it with `git pull upstream main` (the history is full of "Merge remote-tracking branch 'upstream/main'" commits).
- `origin` = `HsuKaoPang/CSCI5942_Assignment`. The student's own work goes here.

Work is committed straight to `main` on top of the upstream releases. When upstream edits a file the student has also changed (READMEs, stubs), merge it by hand.

`Mega assignment/` is the final project. It is a **separate git repo** with its own `CLAUDE.md`, and it is gitignored here. Use that file whenever you work inside it.

## Layout

Each `assignment-N/` directory stands on its own: a `README.md` writeup, starter code with stubs, `tests/`, `configs/*.json`, `slurm/*.sbatch`, and a LaTeX `report.tex`/`report.pdf`. A0 is just `assignment_0.md` at the root. `Latex_Template.tex` at the root is the report template.

| Dir | Topic | Main files |
| --- | --- | --- |
| `assignment-1` | GPT from scratch on Shakespeare, scaling ladder | `model.py`, `train.py`, `sample.py`, `fit_scaling_law.py`, `configs/{smoke,pico,nano,micro,mini,base}.json` |
| `assignment-2` | Evaluation with Opik and an LLM judge (Gemini) | `eval.py`, `judge.py` (`RUBRICS`, `--mock`), `protocol.py`, `results/` |
| `assignment-3` | Common Crawl data pipeline with PySpark | `pipeline/build_corpus.py` (student), `pipeline/catalog.py` (`MIXES`), `data/{fetch,loader}.py`, `eval/` |
| `assignment-4` | Distributed training: DDP, TP, PP, hybrid | `bench/allreduce.py`, `slurm/*_a40.sbatch`, `logs/`, `quiz_prep/` |
| `assignment-5` | PEFT vs full fine-tuning (GPT-2 on E2E) | `lora.py`, `sft.py` (template and loss mask), `train.py`, `eval.py`, `plot_knee.py` |
| `assignment-6` | Serving GPT-2: KV cache, PagedAttention, continuous batching | `model.py` (KVCache), `paged.py`, `engine.py` (Scheduler), `bench_kv.py`, `bench_serve.py` |
| `assignment-7` | RLAIF: GRPO with a Gemini judge as the reward | `grpo.py` (four fenced blocks are Part 2), `judge.py` (`RUBRICS`), `train_sft.py`, `eval.py`, `plot_run.py` |

Later assignments reuse earlier code. Each README has a "What comes from where" table. A2 evaluates A1 checkpoints through relative paths such as `../assignment-1/out/base`. `model.py` in A5–A7 is the A1 GPT with pretrained GPT-2 weight loading added. The A7 `sft.py` is A5's, already filled in. A README marking a file "do not edit" means the grader depends on it.

## Environment

- Python 3.12. The shared venv is `.venv/` at the repo root, and it is the default (use `../.venv/bin/python` from inside an assignment). Some assignments have their own `.venv` (A2, A3, A5). If a directory has its own `.venv`, use that one.
- Each assignment from A3 on has a `requirements.txt` and a `check_env.py`.
- **pytest:** globally installed ROS plugins break pytest collection. Run the tests like this:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
  ```
  A1 is the exception: its `pytest.ini` disables the `launch_testing`/`launch_ros` plugins. The A4 tests run on CPU with a synthetic corpus. The A3 tests start a local SparkSession (this needs Java).
- Every command assumes `cwd` is the assignment directory, because the scripts use paths like `configs/...`, `out/...` and `data/...`.
- Before any stub is implemented, its tests fail with `NotImplementedError`. That is expected.

## Compute

- **Local GPU:** a laptop RTX A1000 with only 4 GB. It is enough for A1-scale models and for smoke runs with `--model gpt2`. It **cannot** fit gpt2-xl (A6 benchmarks) or a full fine-tune of gpt2-medium with AdamW (A5 FFT, A7 GRPO).
- **NCSA Delta** runs the real jobs. The sbatch scripts use `--account=bidb-delta-gpu --partition=gpuA40x4`, run `module load pytorch-conda/2.12`, then `conda activate base`, then `source .venv/bin/activate`, and write logs to `logs/%x-%j.out`. Submit with `sbatch slurm/<script>.sbatch [args...]`; the extra args are passed to the Python script. Slurm logs from finished runs are committed under `assignment-*/logs/`.
- **LLM judge (A2, A7):** `gemini-2.5-flash` via `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Every script takes `--judge mock` (free and deterministic). Use it to test the plumbing before spending credit. A7 caches judge calls in `out/judge-cache.json`.

## Typical commands (per assignment README)

```bash
# A1
python data/prepare.py && python train.py --config configs/smoke.json
python sample.py --run out/base --prompt "DUKE OF" --tokens 400

# A5 / A7 data
python data/prepare_e2e.py

# A7
python judge.py --part1 --judge mock
python grpo.py --config configs/judge-leash.json --judge mock --steps 2
python eval.py --run out/judge-leash --judge gemini
python plot_run.py out/judge-leash
```

## Outputs and gitignore

`out/` (checkpoints, run logs) and `results/` are produced by the scripts. Some of them are gitignored per assignment, along with `*.ckpt`, `ckpt.pt`, `*.parquet`, Spark output and `*.warc.wet.gz`. Large data is regenerated with `data/prepare*.py` or `data/fetch.py`. Some assignment dirs also contain LaTeX build artifacts (`*.aux`, `*.fls`, `*.synctex.gz`, ...) and submission zips (`*_bill_hsu.zip`).

## Working on graded code

These are graded assignments under the course honor code and AI-use policy (see the course logistics page linked in `README.md`).

- Stubs (`raise NotImplementedError`, "fenced" blocks, sections marked YOURS) are the student's work. Do not fill them in unless asked. Explain the code or guide the student instead.
- Do not use instructor solutions or reference results if any show up in a release.
- Only push to `origin`, never to `upstream`.
