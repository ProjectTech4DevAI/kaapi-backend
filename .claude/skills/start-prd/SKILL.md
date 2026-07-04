---
name: start-prd
description: Synthesize the current conversation into a product-level PRD (the why/what, not the how) at features/<feature>/PRD.md — no interview.
disable-model-invocation: true
---

Synthesize the PRD from the current conversation and your codebase understanding. Do NOT interview the user — use what you already know.

The PRD is a **product spec**. It answers one question: *why are we doing this and what does success look like?* It is written for everyone — PM, design, engineering, stakeholders — in user and business terms.

**Hard boundary — the PRD does NOT contain** databases, APIs, algorithms, infrastructure, code structure, or how anything is built. If you find yourself naming a table, an endpoint, or a module, stop: that belongs in the SRD (Software Requirements Doc), not here.

## Process

1. Explore the repo to understand the current state of the codebase, if you haven't already. Use domain vocabulary from the codebase throughout the PRD.

2. Draft the PRD using the template below, keeping every section strictly product-level. Before finalizing, check with the user that the **Goals**, **Non-Goals**, and **which stories are must-have** match their expectations — these three are where misalignment hides.

3. Write the finished PRD to `features/<feature-slug>/PRD.md`, where `<feature-slug>` is a short kebab-case slug (e.g. `features/account-balances/PRD.md`); create the directory if it doesn't exist. If an `SRD.md` already exists for this feature (from the `srd-creator` skill), reuse its slug exactly and write `PRD.md` beside it — don't create a parallel folder. This file is the deliverable, referenced later when building the SRD and Engineering Plan.

<prd-template>

## Problem Statement

The user pain or business need, from the user's perspective. Why does this matter, and to whom?

## Users / Personas

Who this is for — the actors who will use or be affected by the feature.

## Goals

The desired outcomes — what we want to be true once this ships. State them qualitatively; the numbers live in Success Metrics.

## Non-Goals

What is explicitly out of scope. Drawing this line prevents scope creep and misaligned reviews.

## Success Metrics

How we'll know it worked — the KPIs or targets that should move if this succeeds.

## User Stories / Use Cases

A numbered list of user stories, ordered must-have first, each tagged `[must-have]` or `[nice-to-have]`. Every Persona above should trace to at least one story, and every Goal to at least one. Format:

1. `[must-have]` As an <actor>, I want a <feature>, so that <benefit>

<user-story-example>
1. `[must-have]` As a mobile bank customer, I want to see balance on my accounts, so that I can make better informed decisions about my spending
</user-story-example>

## UX / Flows

The experience at a high level — key user flows, screens, and interactions. Link mockups if they exist. Describe what the user sees and does, not how it's implemented.

</prd-template>
