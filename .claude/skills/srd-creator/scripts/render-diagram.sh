#!/usr/bin/env bash
# Render a mermaid source to a high-quality png for an SRD.
# Usage: render-diagram.sh <input.mmd> <output.png>
# No global installs: uses npx for mermaid-cli.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <input.mmd> <output.png>" >&2
  exit 1
fi

SRC="$1"
OUT="$2"
mkdir -p "$(dirname "$OUT")"

# High-res render (wide + scaled) keeps sequence-diagram text crisp.
npx -y -p @mermaid-js/mermaid-cli mmdc -i "$SRC" -o "$OUT" -w 1600 -s 3 -b white

echo "wrote $OUT"
