#!/bin/sh
set -eu

upstream_url='https://github.com/DragonOS-Community/DragonOS.git'
source_commit="${1:-45931ee3b3e66892533563f73023021a83f89b2d}"
source_prefix='kernel/crates/another_ext4'
tracking_branch='sync/dragonos-monorepo'
upstream_ref='refs/heads/master'
temporary_clone="$(mktemp -d)"

cleanup() {
    rm -rf "$temporary_clone"
}
trap cleanup EXIT HUP INT TERM

if test -n "$(git status --porcelain)"; then
    printf '%s\n' 'refusing to sync with a dirty worktree' >&2
    exit 1
fi

git remote get-url dragonos-monorepo >/dev/null 2>&1 || \
    git remote add dragonos-monorepo "$upstream_url"
git clone --filter=blob:none --no-checkout "$upstream_url" "$temporary_clone/dragonos"
git -C "$temporary_clone/dragonos" fetch --no-tags origin "$upstream_ref"
git -C "$temporary_clone/dragonos" cat-file -e "$source_commit^{commit}"
git -C "$temporary_clone/dragonos" cat-file -e "$source_commit:$source_prefix"
git -C "$temporary_clone/dragonos" checkout --detach "$source_commit"

split_commit="$(git -C "$temporary_clone/dragonos" subtree split --prefix="$source_prefix" "$source_commit")"
git fetch --no-tags "$temporary_clone/dragonos" "$split_commit:refs/heads/$tracking_branch"

printf '%s\n' "source_commit=$source_commit"
printf '%s\n' "source_prefix=$source_prefix"
printf '%s\n' "split_commit=$split_commit"
