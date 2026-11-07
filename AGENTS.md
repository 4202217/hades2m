# hades2m agent notes

This repository is a fork of nxnative (Switch games running natively on Apple platforms)
carrying the Hades II port. The game profile lives in `nxnative/dm/games/0100A00019DE0000/`;
the one-command IPA entry point is `make_ipa.sh` at the repository root (it wraps
`nxnative/scripts/build_ipa_from_nsp.py`: extract → rewrite → shader libraries → unsigned IPA).

- Users provide `games/` (NSP/NSZ) and `keys/prod.keys` themselves. Neither, nor `docs/`,
  `.audit/`, `survey/`, `work/`, `tools/`, `venv/`, is committed (see `.gitignore`); do **not**
  copy their contents (paths, values, device info) into replies, commits or documentation.
- Build verification: `./make_ipa.sh` produces `games/HadesII.ipa` (unsigned). Acceptance
  checks: the romfs marker (`Scripts/AchievementData.lua` inside the app's romfs), no
  `_CodeSignature`, the icon present in Assets.car, and `CFBundleDisplayName` taken from the
  NSP control data. A successful build alone does not prove the game plays.
- `nxnative/` is a git subtree (upstream seedds/nxnative): never move or reshuffle its
  internals; sync with `git subtree pull --prefix=nxnative <remote> main`.
- `nxnative/dm/host/gen/<id>/game_profile.h` is generated from
  `nxnative/dm/games/<id>/game.tsv`; after editing a profile just re-run the build — never
  hand-edit generated files.
- This repository has no signing/install scripts: install and verify with a self-signing tool
  (AltStore / Sideloadly), or build a signed variant yourself following
  `nxnative/docs/development.md`.
