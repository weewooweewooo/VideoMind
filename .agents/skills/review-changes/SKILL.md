---
name: review-changes
description: Review VideoMind working-tree changes before commit when the user asks for a final review, pre-commit audit, or regression assessment; review first and do not implement fixes unless separately requested.
---

# Review VideoMind changes

Perform a review of the current working tree, not a general development pass.

## Review

1. Read the applicable `AGENTS.md`, then capture `git status --short` and identify staged, unstaged, and untracked changes. Treat pre-existing changes as part of the review scope unless the user narrows it.
2. Inspect the complete relevant diff against `HEAD`, including staged changes. Read untracked files directly because Git does not include them in ordinary diffs. Check the final combined file state, not only individual diff fragments.
3. Read enough surrounding code and callers to understand each change before judging it. Follow affected ingestion, chunking, retrieval, evaluation, cache, and CLI contracts when relevant.
4. Look for concrete correctness defects, regressions, edge cases, accidental behavior changes, duplication, unjustified abstractions, and unrelated modified files. Verify that the implementation follows `AGENTS.md` and preserves deterministic retrieval and public contracts where applicable.
5. Run the smallest useful existing checks for the files and behavior changed. For retrieval changes, use the repository evaluator when its required fixture is available. Run `git diff --check`. Do not add dependencies or trigger expensive transcription merely to broaden validation.

Initially review only. Do not edit, format, stage, discard, or otherwise modify the working tree unless the user separately asks for fixes.

## Report

- Put actionable findings first, ordered by severity. Give the file path and the narrowest useful location, explain the failure mode, and distinguish actual defects from optional improvements.
- Then report validation performed and its result, followed by anything that could not be validated and why.
- Note unexpected or unrelated changed files.
- If there are no meaningful findings, state that explicitly. Do not invent findings to fill the report.
