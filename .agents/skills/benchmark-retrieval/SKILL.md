---
name: benchmark-retrieval
description: Run or assess controlled VideoMind retrieval experiments when the user asks to benchmark, compare, evaluate, or decide whether to keep a retrieval change using the repository's current evaluator; do not use for ordinary feature development.
---

# Benchmark VideoMind retrieval

Use the repository's live evaluator and production retrieval path. Do not substitute historical commands, datasets, metrics, or thresholds.

## Establish the comparison

1. Read `AGENTS.md`, `evaluate.py`, `evaluation.json`, `src/retrieval.py`, and the live chunking/configuration code before running anything. Capture `git status --short` and identify the single experimental variable.
2. Record the source revision and working-tree state; evaluation file identity and video fixture identity; transcript/cache provenance; chunking and ASR configuration; and retriever tokenization, stemming, stopword, score-filtering, ordering, and top-k behavior. Record values from the live files rather than copying values from this skill.
3. Freeze `evaluation.json`, its expected windows and query types, the video/transcript fixture, and every non-experimental setting. A fresh transcription is a different fixture: disclose it and do not attribute its metric movement solely to retrieval.
4. Before applying the candidate, run the current implementation as the baseline with `py -3.13 evaluate.py` and preserve the complete output. The `before_migration_baseline` metadata in `evaluation.json` is historical context, not a substitute for a comparable live baseline. If the candidate already exists, use only a preserved baseline produced with the same dataset and fixture; otherwise the comparison is `INCONCLUSIVE` until a valid baseline can be run.

## Run and assess

1. Change only the user-authorized experimental variable. Do not silently tune retrieval parameters, labels, expected windows, queries, ASR output, or rejection behavior between runs.
2. Run the candidate with the same `py -3.13 evaluate.py` command and unchanged inputs. Preserve each completed command's full output before starting another variant. If interrupted, keep completed results, mark the in-flight run incomplete, and never estimate missing metrics.
3. Compare only metrics emitted by the current evaluator. At present, assess positive-query Rank-1, Top-3, MRR, and per-query ranks separately from negative-query false positives and per-query rejection outcomes. Treat any additional timing or diagnostics as measured evidence only when actually collected, not as evaluator output.
4. Identify aggregate improvements, aggregate regressions, and individual query movements. Check that claimed gains are not caused by dataset, transcript, configuration, or unrelated code changes. Consider implementation and dependency cost without letting complexity override measured quality.
5. Do not modify production code, evaluation data, or retrieval settings merely to improve the result. Do not retain experimental code or select a production default unless the user authorized that action.

## Recommendation

End with exactly one evidence-backed recommendation:

- `KEEP`: the candidate provides a meaningful measured benefit without an unacceptable regression or scope violation.
- `REJECT`: measurements regress or fail to justify the candidate's cost or complexity.
- `INCONCLUSIVE`: the run, baseline, fixture control, or evidence is incomplete or non-comparable.

Report baseline and candidate configurations, exact supported metrics, positive and negative behavior separately, per-query regressions that affect the decision, completed validation, and any limitations.
