# PRD: Native LLM-as-a-Judge Correctness Score

**Status:** Draft
**Date:** 2026-06-24
**Owner:** Eval platform

---

## Problem Statement

Today, a Kaapi evaluation gives you a *cosine similarity* score — how *similar* a generated answer is to the ground truth — but not whether it is *correct*. Getting a correctness judgment requires an LLM-as-a-judge, and the only way to get one is to manually configure a model-based evaluator inside the third-party Langfuse dashboard — a per-project, out-of-platform setup step that every team has to repeat by hand.

This breaks the promise of an end-to-end evaluation experience:

- **Eval owners** can't judge answer correctness using an LLM without leaving Kaapi and learning a separate tool.
- **Setup is manual and per-project**, so correctness judging should be customized based on the NGO's bot context and answer samples for better correctness scores — but today it is easily forgotten and effectively unavailable to teams who don't know Langfuse.
- **The judging logic lives outside Kaapi**, so teams can't tailor it (their own rating examples, their own model choice) without engineering involvement, and Kaapi has no record of how a row was judged.

We want correctness judging to be a native, automatic, and tailorable part of running an evaluation — so that every evaluated row comes back with both a similarity score *and* a correctness score, with no Langfuse setup required.

## Users / Personas

- **Eval owner** — the person who runs an evaluation on their project and reads the results. Wants a trustworthy correctness signal without leaving Kaapi or touching Langfuse.
- **Per-project team / project admin** — wants to tailor how the judge rates answers for their domain (add their own rating examples, pick the model and its settings) without filing an engineering request or waiting for a deploy.
- **Engineering / platform team** — wants judging logic owned inside Kaapi (versionable, observable, supportable) rather than scattered across external dashboards.

## Goals

1. An evaluation produces an **LLM-as-a-judge correctness score natively inside Kaapi**, with zero manual Langfuse configuration.
2. Every evaluated row ends with **two scores**: the cosine similarity score and the new correctness (judge) score.
3. The judge is **reference-based**: it compares the generated answer against the ground truth (and the original question) and returns a 0–1 correctness score plus a short reasoning explanation.
4. Judging is **automatic** — it runs as part of an evaluation, with no new trigger, flag, or opt-in.
5. The judge works **out of the box with no configuration**, using a built-in default prompt and a fallback model.
6. A project team can **tailor the judge themselves** — custom rating examples, model choice, and model settings — and have it **take effect on the next run without a deploy**.
7. Both scores are **visible in Kaapi's results and in the Langfuse traces** (Langfuse remains the synced system of record).

## Non-Goals

Explicitly out of scope for this iteration:

- **Batch evaluations.** Only fast evaluations are covered. Batch-mode judging is deferred.
- **Removing Langfuse.** Results continue to sync to Langfuse; this work does not replace or retire it.
- **Error and failure handling.** Robust behavior when a judge call fails or returns a malformed result is deferred to a later pass.

## Success Metrics

- **Zero-config coverage:** A project with no judge configuration still gets *both* a similarity score and a correctness score on every evaluated row, using the default prompt and fallback model. Target: 100% of evaluated rows in a fast evaluation receive both scores.
- **Self-service tailoring:** A project can save its own rating examples and model choice, and subsequent runs demonstrably use them — with no deploy and no engineering ticket.
- **Explainability:** Every judge result includes a score *and* a reasoning explanation. Target: 100% of judged rows carry a reasoning string.
- **Persistence & visibility:** Both scores are persisted by Kaapi and both appear on each row's Langfuse trace. Target: every evaluated row's trace shows two distinctly-labeled scores.
- **Reversibility:** Removing a project's judge configuration cleanly reverts that project to the default prompt + fallback model on the next run.
- **Acceptable run time:** Adding the judge does not push a fast evaluation outside its acceptable run-time window for the supported row count. (Target window to be confirmed — see Open Questions.)

## User Stories / Use Cases

1. As an eval owner, I want each evaluated row to come back with a correctness score in addition to the similarity score, so that I can tell whether the answer was *right*, not just *close*.
2. As an eval owner, I want the correctness score to come with a short reasoning explanation, so that I understand *why* a row was judged the way it was.
3. As an eval owner, I want judging to happen automatically when I run an evaluation, so that I don't have to remember a separate step or flip a setting.
4. As an eval owner on a brand-new project with no setup, I want to still get a correctness score, so that the feature works for me out of the box without any configuration.
5. As a project admin, I want to add my own rating examples to guide the judge, so that the judge rates answers the way my domain expects.
6. As a project admin, I want to choose which model does the judging and adjust its settings (e.g. temperature), so that I can balance quality, cost, and consistency for my project.
7. As a project admin, I want to view my project's current judge configuration, so that I can see what is currently in effect.
8. As a project admin, I want to update my judge configuration and have the change take effect on the very next run, so that I can iterate without waiting for a deploy.
9. As a project admin, I want to delete my judge configuration, so that my project falls back to the platform default prompt and model.
10. As an eval owner, I want both scores to appear on the Langfuse trace for each row, so that my existing Langfuse-based reporting continues to work and shows correctness alongside similarity.
11. As an eval owner, I want both scores persisted with the evaluation results, so that I can re-open a past run and still see the correctness judgments.
12. As an eval owner reviewing results, I want the two scores clearly distinguished (which is cosine similarity, which is correctness), so that I don't confuse them.
13. As a platform engineer, I want the judging logic and its default prompt owned inside Kaapi, so that it is consistent across projects and supportable without per-project Langfuse work — and so eval owners never have to open or learn the Langfuse dashboard.

## UX / Flows

**Running an evaluation (the default, no-config path)**
An eval owner runs a fast evaluation exactly as they do today — same trigger, no new options. When results come back, each row now shows **two scores**: the cosine similarity score and a new correctness score, each clearly labeled. Opening a row reveals the judge's short reasoning for its correctness score. The same two scores are visible on that row's Langfuse trace. The owner did nothing extra and configured nothing.

**Tailoring the judge (the self-service config path)**
A project admin opens their project's judge configuration. From there they can:
- Add, edit, or remove **custom rating examples** that show the judge how their team grades answers.
- Choose the **model** that performs judging and adjust its **settings** (e.g. temperature).
- **Save** the configuration. The next evaluation run uses it automatically — no deploy.
- **View** the configuration at any time to see what's currently in effect.
- **Delete** the configuration to revert the project to the built-in default prompt and fallback model.

**What a judged row contains**
For every evaluated row: the original question, the generated answer, the ground-truth answer, a similarity score (0–1), and a correctness score (0–1) with a short reasoning explanation — all persisted and all reflected in the Langfuse trace.

*(Exact screen layouts, the input format for custom rating examples, and the precise score labels shown to users are to be finalized — see Open Questions.)*

## Scope & Priorities

**Must-have (this release)**
1. Automatic, native correctness judging on fast evaluations — no new trigger.
2. Two scores per evaluated row (similarity + correctness), both persisted and both shown on the Langfuse trace.
3. A judge result that includes both a 0–1 score and a reasoning explanation.
4. Working default behavior with no configuration (built-in prompt + fallback model).
5. Per-project judge configuration the team can create, view, update, and delete — covering custom rating examples, model choice, and model settings — effective on the next run without a deploy.
6. Deleting the configuration reverts the project to the default prompt + fallback model.

**Nice-to-have / later**
- Robust judge error and retry handling (deferred this iteration).
- Confirmed performance budget and row-count guidance for the added per-row judge call.
- Refined UX for entering custom rating examples and for score labeling in Langfuse.

**Explicitly later (Non-Goals above):** batch-mode judging, removing Langfuse.

## Open Questions

1. **Error & retry behavior** when a judge call fails or returns a malformed result (deferred this iteration, but needs a defined fallback so a failed judge doesn't block the similarity score or the run).
2. **Performance budget** — the judge adds one model call per row; confirm acceptable fast-evaluation run times and the supported row count.
3. **Custom rating examples format** — the exact shape and entry UX for the examples a team provides.
4. **Score labeling** — the precise names shown to users for the two scores, in Kaapi results and in Langfuse traces.
