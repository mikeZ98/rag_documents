# shellcheck shell=sh
# Git identity invariant (see CLAUDE.md), shared by pre-commit, commit-msg and pre-push.
#
# Every commit is authored AND committed by the canonical identity below, so the
# history maps onto one GitHub profile. Message content is not policed here:
# trailers such as Co-Authored-By: are allowed. Sourced, not run.

CANON_NAME="mikeZ98"
CANON_EMAIL="145975972+mikeZ98@users.noreply.github.com"

reject() {
  echo "identity guard: REJECTED — $1" >&2
  echo "identity guard: commits must be authored and committed by $CANON_NAME <$CANON_EMAIL>." >&2
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

# Audit already-recorded commits: cherry-pick and rebase replay commits without
# running pre-commit, so pre-push and CI re-verify every commit in a revision range.
check_revisions() {
  for sha in $(git rev-list "$@"); do
    for role in a c; do
      name=$(git log -1 --format="%${role}n" "$sha")
      email=$(git log -1 --format="%${role}e" "$sha")
      [ "$name" = "$CANON_NAME" ] && [ "$email" = "$CANON_EMAIL" ] ||
        reject "commit $sha has $role-ident '$name <$email>'"
    done
  done
}
