# shellcheck shell=sh
# Git attribution invariant (see CLAUDE.md), shared by pre-commit and commit-msg.
#
# Every commit is authored AND committed solely by the canonical identity below,
# and its message carries no co-author or AI-generation metadata. Sourced, not run.

CANON_NAME="mikeZ98"
CANON_EMAIL="145975972+mikeZ98@users.noreply.github.com"

# Case-insensitive ERE, one alternative per line of intent:
#   trailers that credit someone else or a tool, AI tool / vendor fingerprints,
#   and the robot emoji used by "Generated with" footers.
FORBIDDEN_MESSAGE_RE='^[[:space:]]*co-authored-by[[:space:]]*:'
FORBIDDEN_MESSAGE_RE="$FORBIDDEN_MESSAGE_RE|^[[:space:]]*(ai|assisted|generated|made)-(by|with)[[:space:]]*:"
FORBIDDEN_MESSAGE_RE="$FORBIDDEN_MESSAGE_RE|generated (with|by)|noreply@anthropic\\.com|anthropic"
FORBIDDEN_MESSAGE_RE="$FORBIDDEN_MESSAGE_RE|claude code|claude\\.ai|claude\\.com|chatgpt|openai codex|copilot"
FORBIDDEN_MESSAGE_RE="$FORBIDDEN_MESSAGE_RE|🤖"

reject() {
  echo "attribution guard: REJECTED — $1" >&2
  echo "attribution guard: commits must be by $CANON_NAME <$CANON_EMAIL> with no AI/co-author metadata." >&2
  exit 1
}

# `git var` resolves the ident exactly as `git commit` will record it, including
# --author, GIT_AUTHOR_* / GIT_COMMITTER_* overrides and every config layer.
check_identity() {
  for role in AUTHOR COMMITTER; do
    ident=$(git var "GIT_${role}_IDENT") || reject "cannot resolve the $role identity"
    # Strip the trailing "<unix-time> <tz>" stamp.
    ident=${ident% * *}
    name=${ident% <*}
    email=${ident##*<}
    email=${email%>}
    [ "$name" = "$CANON_NAME" ] || reject "$role name is '$name', expected '$CANON_NAME'"
    [ "$email" = "$CANON_EMAIL" ] || reject "$role email is '$email', expected '$CANON_EMAIL'"
  done
}

# Only the message body is inspected: `git commit -v` appends the diff below a
# scissors line, and comment lines are stripped by git before recording.
check_message() {
  body=$(sed -e '/^# -\{24\} >8 -\{24\}$/,$d' -e '/^#/d' "$1")
  hit=$(printf '%s\n' "$body" | grep -inE "$FORBIDDEN_MESSAGE_RE" | head -n 1)
  [ -z "$hit" ] || reject "forbidden attribution metadata in the message (line $hit)"
}

# Audit already-recorded commits: cherry-pick and rebase replay commits without
# running pre-commit, so pre-push and CI re-verify every commit in a revision range.
check_revisions() {
  msg_file=$(mktemp) || reject "cannot create a temporary file"
  for sha in $(git rev-list "$@"); do
    for role in a c; do
      name=$(git log -1 --format="%${role}n" "$sha")
      email=$(git log -1 --format="%${role}e" "$sha")
      [ "$name" = "$CANON_NAME" ] && [ "$email" = "$CANON_EMAIL" ] ||
        { rm -f "$msg_file"; reject "commit $sha has $role-ident '$name <$email>'"; }
    done
    git log -1 --format=%B "$sha" >"$msg_file"
    (check_message "$msg_file") || { rm -f "$msg_file"; reject "commit $sha message"; }
  done
  rm -f "$msg_file"
}
