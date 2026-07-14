#!/bin/bash
# Stop hook: structural code change without a docs/wiki update -> block stop with a
# one-line reminder. Silent (0 tokens) in every other case. Detector stays dumb on
# purpose: file->wiki-page mapping lives in docs/wiki/INDEX.md, Claude resolves it.
INPUT=$(cat)
grep -Eq '"stop_hook_active"[[:space:]]*:[[:space:]]*true' <<<"$INPUT" && exit 0
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
BASE=$(git merge-base origin/main HEAD 2>/dev/null || echo HEAD)
CHANGED=$({ git diff --name-only "$BASE" 2>/dev/null; git status --porcelain -uall 2>/dev/null | awk '{print $NF}'; } | sort -u)
STRUCT=$(grep -E '^backend/app/(api/routes|models|services|crud|celery/tasks)/' <<<"$CHANGED")
[ -z "$STRUCT" ] && exit 0
grep -q '^docs/wiki/' <<<"$CHANGED" && exit 0
FILES=$(head -5 <<<"$STRUCT" | tr '\n' ' ' | tr -d '"\\')
printf '{"decision":"block","reason":"Structural change without wiki update: %s— update the matching docs/wiki/modules/*.md page (names+paths only, via docs/wiki/INDEX.md), and docs/wiki/domain-map.md if entities/edges changed. If no update is needed, state why and stop."}\n' "$FILES"
