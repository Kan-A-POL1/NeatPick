#!/usr/bin/env bash
# Apply NeatPick's changes to a Physical AI Studio checkout of the upstream junction2026 branch.
# Usage: apply.sh <physical-ai-studio checkout>     (run again after every `uv sync`: it re-patches the venv packages)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO="${1:?usage: apply.sh <physical-ai-studio checkout>}"
BASE=285924d9381238d3ef2478294c2b85a0eb6ac62b
SITE="$STUDIO/application/backend/.venv/lib/python3.13/site-packages"

if git -C "$STUDIO" apply --reverse --check "$HERE/patches/physical-ai-studio.patch" 2>/dev/null; then
  echo "Studio patch: already applied"
else
  [ "$(git -C "$STUDIO" rev-parse HEAD)" = "$BASE" ] || echo "warning: checkout is not junction2026@${BASE:0:8}" >&2
  git -C "$STUDIO" apply "$HERE/patches/physical-ai-studio.patch"
  echo "Studio patch: applied"
fi

[ -d "$SITE" ] || { echo "no backend venv at $SITE: run 'uv sync' in application/backend, then this script again" >&2; exit 1; }
for patch in so101-telemetry capture-depth capture-selfheal openvino-gpu-f32-fallback; do
  file="$HERE/patches/physicalai/$patch.patch"
  if (cd "$SITE" && patch --dry-run --silent -R -p1 < "$file" >/dev/null 2>&1); then
    echo "$patch: already applied"
  else
    (cd "$SITE" && patch --forward --silent -p1 < "$file")
    echo "$patch: applied"
  fi
done
