# PRD: AI-Assisted Prompt Improvement from Evaluation Results

**Status:** Draft · **Owner:** AkhileshNegi · **Date:** 2026-06-24

## Problem Statement

When a team runs an evaluation against a configuration (a prompt + model + knowledge base), the results tell them *how well* the prompt performed — average similarity, correctness, weak categories, and which questions the assistant answered poorly. But turning those results into a *better prompt* is entirely manual today. A user has to read through the failing questions, spot the recurring patterns, reason about what the prompt is missing, and hand-write a new prompt iteration.

That work is slow, requires prompt-engineering skill, and doesn't scale across many configurations or frequent evaluation cycles. The evaluation already produces exactly the evidence needed to improve the prompt — that evidence just sits unused. We want to close the loop: let the user turn a finished evaluation into a concrete, improved prompt iteration with one action.

## Users / Personas

- **Prompt/Config owner** — the person responsible for a configuration's quality. They run evaluations and want their prompt to keep improving without manual rewriting.
- **Non-expert builder** — someone who can run an evaluation but isn't a prompt-engineering specialist. They need help translating "these answers are weak" into "here's a better prompt."
- **Reviewer / approver** — someone who looks at the resulting prompt iterations and decides whether to adopt one, relying on a clear record of where each iteration came from.

## Goals

- Let a user generate an improved prompt iteration directly from a completed evaluation run, in a single action.
- Ground the improvement in the evaluation's own evidence — the questions that performed poorly *consistently*, and the broader patterns across question categories.
- Let the user control what "poor performance" means: which quality metric to judge on, and the cutoff for "too low."
- Produce a new prompt iteration that is clearly marked as AI-generated and traceable back to the evaluation it came from.
- Keep every prior prompt iteration intact — the new one is added alongside, never overwrites.
- Preserve everything about the configuration except the prompt itself (same model, same knowledge base, same other settings), so the improvement is an apples-to-apples prompt change.

## Non-Goals

- **No automatic triggering.** The improvement does not fire on its own when an evaluation completes; the user always opts in per evaluation run.
- **No knowledge-base diagnosis or fixes in this release.** Stale or irrelevant knowledge-base content is a real cause of poor answers, but diagnosing it (and certainly fixing it) is out of scope for v1. The improvement only changes the prompt.
- **No changes to model, knowledge base, or other configuration settings** — only the prompt text changes.
- **No automatic re-evaluation** of the newly generated prompt iteration.
- **No background/async processing or completion notifications** in v1 — the action runs and returns its result directly. (A faster, async version is a likely follow-up.)

## Success Metrics

- **Adoption:** % of completed evaluations on which users invoke the improvement action.
- **Acceptance:** % of AI-generated prompt iterations that users keep / promote rather than discard.
- **Quality lift:** for adopted AI-generated prompts, the change in the chosen quality metric (e.g. average similarity or correctness) when the new prompt is re-evaluated, vs. the prompt it replaced.
- **Effort saved:** reduction in time from "evaluation finished" to "improved prompt exists," vs. the manual baseline.

## User Stories / Use Cases

1. As a config owner, I want to generate an improved prompt from a completed evaluation in one action, so that I don't have to hand-write a new prompt myself.
2. As a config owner, I want the improvement to be based on the questions that scored low, so that the new prompt targets real weaknesses rather than guesses.
3. As a config owner, I want "low scoring" to mean *consistently* low — not a one-off bad answer — so that the improvement focuses on genuine problems and ignores noise.
4. As a config owner, I want the improvement to also account for whole categories of questions that underperform, so that systemic gaps in the prompt are addressed.
5. As a config owner, I want to choose which quality metric defines "low" (similarity-based or judged), so that the analysis matches how I measure quality for this use case.
6. As a config owner, I want to set the cutoff for "too low," so that I control how aggressive the selection of weak questions is.
7. As a non-expert builder, I want the system to write the improved prompt for me, so that I can improve quality without prompt-engineering expertise.
8. As a config owner, I want the new prompt to keep the same model and knowledge base as the evaluated configuration, so that I'm comparing prompt-to-prompt and nothing else changed.
9. As a config owner, I want the new prompt to appear as the next iteration in the configuration's history, so that it slots naturally into how I already manage versions.
10. As a reviewer, I want each AI-generated iteration clearly labeled as "AI Generated," so that I can tell at a glance which prompts a human wrote and which the system proposed.
11. As a reviewer, I want to know which evaluation run an AI-generated prompt came from, so that I can trace the rationale and judge whether to trust it.
12. As a reviewer, I want a short explanation of what the improvement targeted, so that I understand the intent behind the new prompt before adopting it.
13. As a config owner, I want all my previous prompt iterations preserved when a new one is generated, so that I never lose a working prompt and can always revert.
14. As a config owner, I want to generate an improvement only from a finished evaluation, so that the analysis is based on complete results.
15. As a config owner, I want to optionally generate more than one improved iteration over time (e.g. after re-evaluating), so that I can iterate repeatedly toward better quality.

## UX / Flows

**Where it lives:** At the evaluation-run level. When viewing a completed evaluation, the user sees an action to improve the prompt. Because the action starts from a specific evaluation, the underlying configuration and prompt are already known — the user never has to pick a config manually.

**Primary flow:**
1. User opens a completed evaluation run and reviews its results (scores, weak questions, category breakdown).
2. User chooses to improve the prompt. They select the quality metric to judge on and the cutoff for "low."
3. The system analyzes the evaluation: it finds the questions that scored low consistently and the categories that underperform, considers the current prompt, and drafts an improved prompt.
4. A new prompt iteration is created in the configuration's history — labeled as the next iteration with an "AI Generated" marker — and presented to the user along with a short note on what it targeted and which evaluation it came from.
5. The user reviews the new iteration, and can adopt it, edit it further, re-evaluate it, or discard it.

**Key experience qualities:**
- One clear action from the evaluation view; minimal choices (metric + threshold).
- The result is a normal prompt iteration the user already knows how to work with — not a separate, special object.
- The "AI Generated" provenance and the link to the originating evaluation are visible, so trust and traceability are built in.

## Scope & Priorities

**Must-have (v1)**
- Generate an improved prompt iteration from a completed evaluation run, on explicit user action.
- User-selectable quality metric (similarity-based or judged) and threshold.
- Selection of weak questions based on *consistent* low performance, plus underperforming categories.
- New iteration keeps model and knowledge base unchanged — prompt only.
- New iteration added to the configuration's history, clearly marked "AI Generated," traceable to the source evaluation, with a short rationale. Prior iterations preserved.

**Nice-to-have (later)**
- Knowledge-base staleness diagnosis (flagging when poor answers stem from retrieved content, not the prompt).
- Faster background processing with a completion notification.
- One-click re-evaluation of the new iteration to immediately measure the lift.
- Generating and comparing multiple candidate prompts at once.
