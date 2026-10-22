#!/bin/sh
# Audit a revision range against the git attribution invariant (used by CI).
#   scripts/check_attribution.sh origin/main..HEAD
set -eu
. "$(dirname "$0")/../.githooks/_attribution.sh"
[ "$#" -gt 0 ] || { echo "usage: $0 <rev-list args>" >&2; exit 2; }
check_revisions "$@"
echo "attribution guard: all commits in '$*' are clean"
