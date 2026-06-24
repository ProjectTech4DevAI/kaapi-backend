---
name: start-prd
description: Turn the current conversation into a product-level PRD and write it to docs/prd-<feature>.md — no interview, just synthesis of what you've already discussed. The PRD covers the product spec only (why / what / for whom); technical "how" belongs in the SRD.
disable-model-invocation: true
---

This skill takes the current conversation context and codebase understanding and produces a PRD. Do NOT interview the user — just synthesize what you already know.

The PRD is a **product spec**. It answers one question: *why are we doing this and what does success look like?* It is written for everyone — PM, design, engineering, stakeholders — in user and business terms.

**Hard boundary — the PRD does NOT contain** databases, APIs, algorithms, infrastructure, code structure, or how anything is built. If you find yourself naming a table, an endpoint, or a module, stop: that belongs in the SRD (Software Requirements Doc), not here. The PRD says *what the product does for the user*; the SRD says *what the software must do for the PRD to be true*.

## Process

1. Explore the repo to understand the current state of the codebase, if you haven't already. Use the project's domain glossary vocabulary throughout the PRD.

2. Draft the PRD using the template below, keeping every section strictly product-level. Before finalizing, check with the user that the **Goals**, **Non-Goals**, and **Scope** match their expectations — these three are where misalignment hides.

3. Write the finished PRD to `docs/prd-<feature>.md`, where `<feature>` is a short kebab-case slug for the feature (e.g. `docs/prd-account-balances.md`). Create the `docs/` directory if it doesn't exist. Giving each PRD its own file keeps prior PRDs preserved rather than overwritten. This file is the deliverable — it can be referenced later when building the SRD and Engineering Plan.

<prd-template>

## Problem Statement

The user pain or business need, from the user's perspective. Why does this matter, and to whom?

## Users / Personas

Who this is for — the actors who will use or be affected by the feature.

## Goals

The desired outcomes, ideally measurable. What we want to be true once this ships.

## Non-Goals

What is explicitly out of scope. Drawing this line prevents scope creep and misaligned reviews.

## Success Metrics

How we'll know it worked — the KPIs or targets that should move if this succeeds.

## User Stories / Use Cases

A LONG, numbered list of user stories that covers all aspects of the feature. Each in the format:

1. As an <actor>, I want a <feature>, so that <benefit>

<user-story-example>
1. As a mobile bank customer, I want to see balance on my accounts, so that I can make better informed decisions about my spending
</user-story-example>

## UX / Flows

The experience at a high level — key user flows, screens, and interactions. Link mockups if they exist. Describe what the user sees and does, not how it's implemented.

## Scope & Priorities

What is must-have vs nice-to-have for this release. Order by priority.

</prd-template>
