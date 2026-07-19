# another_ext4 upstream provenance

## Upstream source

- **Project:** [DragonOS](https://github.com/DragonOS-Community/DragonOS)
- **Repository:** `https://github.com/DragonOS-Community/DragonOS.git`
- **Subtree:** `kernel/crates/another_ext4`
- **Source commit:** `45931ee3b3e66892533563f73023021a83f89b2d`

## Extraction record

- **Extraction commit:** `571b85084fade21f5c26726a78e71356210c4f86`
- **Command:** `git subtree split --prefix=kernel/crates/another_ext4 45931ee3b3e66892533563f73023021a83f89b2d`
- **Tooling:** Git `2.25.1`; `git subtree` from the same Git installation.
- **Published extracted-history refs:** `dragonos` and `sync/dragonos-monorepo`, both at
  `571b85084fade21f5c26726a78e71356210c4f86`.

## Mango fork and consumer contract

- **Fork:** `git@github.com:Mango-Iced-Americano/another_ext4.git`
- **Development branch:** `mango`
- **Consumer pin:** MangoCore must pin the `mango` gitlink to an immutable full commit SHA,
  never to a branch name. Consumers may advance that pin only after recording the selected
  `mango` SHA and verifying its upstream ancestry.
- `sync/dragonos-monorepo` and `dragonos` contain extracted DragonOS history only.
  Mango-specific changes, conflict resolutions, and provenance metadata belong only on `mango`.

## License and local deviations

- **License:** MIT, as declared by the extracted subtree's `LICENSE`; retain upstream license
  and notice files when syncing.
- **Local patches/deviations:** none at this provenance point. The only Mango-side delta is
  this `UPSTREAM.md` metadata commit; runtime code and extracted subtree content are unchanged.

## Reproduction and verification

```sh
git clone https://github.com/DragonOS-Community/DragonOS.git
git -C DragonOS subtree split --prefix=kernel/crates/another_ext4 45931ee3b3e66892533563f73023021a83f89b2d
git ls-remote git@github.com:Mango-Iced-Americano/another_ext4.git \
  refs/heads/dragonos refs/heads/sync/dragonos-monorepo refs/heads/mango
git merge-base --is-ancestor 571b85084fade21f5c26726a78e71356210c4f86 <mango-sha>
```

## Sync ownership

- **Synced at:** `2026-07-19T15:54:03+08:00`
- **Maintainer:** Mango-Iced-Americano
