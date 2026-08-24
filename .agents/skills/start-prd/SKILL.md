---
name: start-prd
description: Synthesize the current conversation into a product-level PRD (the why/what, not the how) at features/<feature>/PRD.md — no interview.
disable-model-invocation: true
---

Synthesize the PRD from the current conversation and your codebase understanding. Do NOT interview the user — use what you already know.

The PRD is a **product spec**. It answers one question: *why are we doing this and what does success look like?* It is written for everyone — PM, design, engineering, stakeholders — in user and business terms.

**Hard boundary — the PRD does NOT contain** databases, APIs, algorithms, infrastructure, code structure, or how anything is built. If you find yourself naming a table, an endpoint, or a module, stop: that belongs in the SRD (Software Requirements Doc), not here.

## Process

1. Understand the current state of the codebase, if you haven't already: start with `docs/wiki/INDEX.md` + `docs/wiki/domain-map.md` and the relevant `docs/wiki/modules/*.md` page(s) — they carry the domain vocabulary the PRD should use. Explore the repo directly only for what the wiki doesn't answer.

2. Draft the PRD using the template below, keeping every section strictly product-level. Before finalizing, check with the user that the **Goals**, **Non-Goals**, and **which stories are must-have** match their expectations — these three are where misalignment hides.

3. Write the finished PRD to `features/<feature-slug>/PRD.md`, where `<feature-slug>` is a short kebab-case slug (e.g. `features/account-balances/PRD.md`); create the directory if it doesn't exist. If an `SRD.md` already exists for this feature (from the `srd-creator` skill), reuse its slug exactly and write `PRD.md` beside it — don't create a parallel folder. This file is the deliverable, referenced later when building the SRD and Engineering Plan.

<prd-template>

## Problem Statement

The user pain or business need, from the user's perspective. Why does this matter, and to whom?

## Goals

The desired outcomes — what we want to be true once this ships. State them qualitatively; the numbers live in Success Metrics.

## Non-Goals

What is explicitly out of scope. Drawing this line prevents scope creep and misaligned reviews.

## Success Metrics

How we'll know it worked — the KPIs or targets that should move if this succeeds.

## Users / Personas

Who this is for — the actors who will use or be affected by the feature.

## User Stories / Use Cases

Group stories under a **persona heading** (one per Persona above). Don't repeat "As a <actor>" on every line — the heading carries it. Order personas primary-first and, within each, must-have-first. Tag every story `[must-have]` or `[nice-to-have]`. Every Persona above should carry at least one story, and every Goal above should trace to at least one. Number stories continuously across the whole section (don't restart per persona). Format:

**As a <persona heading>:**

1. `[must-have]` I want a <feature>, so that <benefit>
2. `[nice-to-have]` I want a <feature>, so that <benefit>

<user-story-example>
**As a mobile bank customer:**

1. `[must-have]` I want to see the balance on my accounts, so that I can make better informed decisions about my spending
2. `[nice-to-have]` I want to filter transactions by category, so that I can see where my money goes
</user-story-example>

## UX / Flows

The experience at a high level — key user flows, screens, and interactions. Link mockups if they exist. Describe what the user sees and does, not how it's implemented.

</prd-template>
