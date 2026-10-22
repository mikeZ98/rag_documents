#!/bin/sh
# Regenerate requirements.txt (hash-pinned, for plain pip) from uv.lock.
#   scripts/export_requirements.sh
# The project itself is left out (it has no hash), so `pip --require-hashes` works.
# uv export does not carry [[tool.uv.index]] entries, so the PyTorch CPU index
# that serves Linux's torch==X+cpu is added here; without it pip cannot find it.
set -eu
cd "$(dirname "$0")/.."
out=requirements.txt
uv export --frozen --all-extras --no-emit-project --format requirements-txt --no-header -o "$out.tmp" >/dev/null
{
  echo "# Exported from uv.lock by scripts/export_requirements.sh — do not edit by hand."
  echo "--extra-index-url https://download.pytorch.org/whl/cpu"
  cat "$out.tmp"
} >"$out"
rm -f "$out.tmp"
echo "wrote $out"
