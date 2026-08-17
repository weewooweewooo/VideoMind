# VideoMind Agent Guide

## Goal

VideoMind is a lightweight, local-first video understanding and retrieval project.

Optimize for:

* clarity,
* retrieval quality,
* useful AI functionality,
* maintainability,
* and strong portfolio value.

Prefer simple, understandable implementations over additional architecture or abstraction.

## Working Style

* Inspect the existing implementation before making changes.
* Follow the current project structure and conventions.
* Keep changes narrowly scoped to the requested task.
* Prefer modifying existing files over creating new ones.
* Preserve existing behavior unless the task explicitly requires changing it.
* Do not rewrite unrelated code while implementing a change.
* Prefer one clear implementation per responsibility.
* When replacing an implementation, remove obsolete or duplicate code once the replacement is verified.
* If multiple reasonable approaches exist, prefer the simplest one that fits the current architecture.
* Work incrementally:

```text
inspect
→ identify the problem
→ make the smallest useful change
→ validate
→ compare behavior
→ continue
```

## Simplicity and Scope

* Do not introduce new layers, services, classes, configuration files, or dependencies unless they solve a real problem.
* Do not add features simply because they are technically interesting.
* Avoid production-scale infrastructure for problems that can be solved locally.
* Do not introduce databases, vector databases, Docker, microservices, Redis, web APIs, or distributed infrastructure unless the project genuinely requires them.
* When code becomes unnecessarily complex, prefer removing or replacing complexity rather than moving it into additional modules.
* Do not split code into many small files purely to reduce line count.

## Python

* Write clear, idiomatic Python.
* Use Ruff for formatting and linting.
* Keep functions focused and readable.
* Prefer explicit code over clever abstractions.
* Add type hints where they improve clarity.
* Avoid unnecessary classes when functions are sufficient.
* Reuse existing helpers before introducing new ones.
* Handle errors explicitly where failure is meaningful.
* Keep comments focused on *why* something is necessary rather than explaining obvious syntax.

## Dependencies

* Prefer the Python standard library or existing project dependencies when practical.
* Prefer maintained libraries over project-owned implementations of solved problems.
* Do not add a dependency only to solve a small problem that can be implemented cleanly with existing tools.
* If a dependency must change, update the project's dependency declaration accordingly.
* Do not silently change dependency versions unrelated to the task.
* Avoid heavyweight ML dependencies unless their measured benefit justifies the installation, memory, and runtime cost.
* Expensive ML dependencies should be loaded lazily where practical.

## Project Structure

Keep the repository small and easy to navigate.

A developer should be able to identify quickly:

```text
where video enters
where transcription happens
where chunks are created
where indexing happens
where retrieval happens
where the final result is produced
```

Rules:

* Prefer modifying existing files over introducing new modules.
* Keep ingestion, retrieval, orchestration, and other existing responsibilities separated according to the current structure.
* Do not move or rename existing files without a concrete reason.
* Do not keep legacy or alternative implementations after a replacement has been verified unless they still serve a real purpose.
* Avoid wrapper classes or helper modules that add indirection without meaningful behavior.

## Retrieval and AI Changes

Treat retrieval quality as something to measure rather than assume.

When changing retrieval, ranking, chunking, transcription, or AI-related behavior:

* Preserve deterministic behavior where practical.
* Keep retrieval logic understandable and inspectable.
* Avoid hidden heuristics unless explicitly justified by measured behavior.
* Do not change thresholds, ranking semantics, chunking defaults, or public behavior incidentally.
* Compare behavior before and after meaningful retrieval changes.
* Prefer evidence from real project inputs over synthetic assumptions.
* Reuse existing evaluation inputs and metrics when comparing retrieval changes.
* Do not silently modify evaluation queries or expected results to make a change look better.
* Change one major retrieval variable at a time so regressions can be attributed correctly.

When comparing approaches such as BM25, TF-IDF, embeddings, hybrid retrieval, or reranking, consider:

* Rank-1 accuracy,
* Top-k accuracy,
* MRR,
* unrelated-query false positives,
* timestamp/window usefulness,
* latency,
* dependency cost,
* implementation complexity.

Do not introduce a more complex retrieval method unless evaluation shows that the simpler approach has a meaningful limitation it solves.

## Evaluation

Keep evaluation small and useful.

* Prefer a small fixed set of real queries over large testing infrastructure.
* Include exact queries, paraphrases, and unrelated negative queries where relevant.
* Use existing evaluation files rather than creating additional benchmark frameworks.
* Treat evaluation data as frozen during before/after comparisons.
* Investigate regressions instead of tuning the evaluation set around them.
* CLI smoke tests are acceptable when formal tests would add more maintenance than value.

Do not create a new test framework, test directory, benchmark suite, or validation infrastructure unless specifically requested.

## CLI and Compatibility

* Preserve existing CLI arguments and output structure unless the requested task explicitly changes them.
* Maintain backward compatibility when reasonably possible.
* Do not introduce another entry point when the existing one can support the feature.
* Keep CLI output deterministic where practical.

## Validation

After making changes:

1. Run Ruff or the relevant formatting/lint checks on changed Python code.
2. Run the smallest relevant CLI or functional smoke check.
3. Exercise the behavior that was changed.
4. For retrieval changes, run the existing evaluation when applicable.
5. Check for obvious regressions in adjacent behavior.
6. Run `git diff --check`.
7. Report what changed and what was validated.

If useful validation cannot be run, state exactly what could not be verified and why.

## Git

* Run `git status` before making meaningful changes.
* Treat existing working-tree changes as the current baseline unless there is clear evidence they are accidental or they conflict with the requested task.
* Do not discard, overwrite, or revert unrelated working-tree changes.
* Do not create commits unless explicitly asked.
* Do not create branches unless explicitly asked.
* Do not push changes unless explicitly asked.
* Before finishing, summarize the files changed.

## Before Finishing

Check that:

* The requested task is actually complete.
* No unnecessary files were created.
* No duplicate or obsolete implementation remains without a reason.
* No unrelated behavior was changed.
* New complexity is justified.
* Relevant formatting and validation were performed.
* Retrieval changes were compared against the existing baseline when applicable.
* The final response is concise and identifies any remaining limitation.
