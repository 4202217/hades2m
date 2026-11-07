# hades2m — Hades II for iOS

A port of the Switch version of *Hades II* running natively on iPhone / iPad, built on
[nxnative](https://github.com/seedds/nxnative) (a framework that runs Switch games natively on
Apple platforms). This repository carries the Hades II game profile and the adaptations
(full-screen widescreen, touch controls, HD haptics), plus a **one-command IPA build**:
bring your own game files and keys, get an unsigned IPA, and install it with a self-signing tool.

## Requirements (macOS)

- [Xcode](https://apps.apple.com/app/xcode/id497799835) from the App Store
- Homebrew packages:

  ```bash
  brew install xcodegen spirv-cross zstd
  brew install --cask dotnet-sdk
  ```

- Python 3: the first `make_ipa.sh` run creates `.venv` automatically and installs
  `lz4 / capstone / fonttools` (plus `nsz` when an NSZ file is present).

## Build the IPA in one command

1. Put **your own** game files in `games/` (base game + update, NSP or NSZ) and your
   `prod.keys` in `keys/`:

   ```
   games/
     Hades II v0.0.126424 [0100A00019DE0000][v0].nsz        # base game
     Hades II v1.0.137848 [0100A00019DE0800][v589824].nsp   # update
   keys/
     prod.keys
   ```

2. Build:

   ```bash
   ./make_ipa.sh
   ```

   The first run takes about 10–35 minutes (mostly shader library compilation); later runs
   take about 10 minutes thanks to the cache. Output: `games/HadesII.ipa` (unsigned). Full
   logs of every step land in `games/0100A00019DE0000_build/logs/`.

3. Install: hand the IPA to a self-signing tool such as
   [AltStore](https://altstore.io), [Sideloadly](https://sideloadly.io) or LiveContainer —
   they swap in your own certificate and bundle ID at install time. Free Apple ID signatures
   expire after 7 days (just re-sign); at most 3 self-signed apps per device.

> Bring your own game and keys — this repository ships no game data and no keys, and
> `.gitignore` keeps `games/` and `keys/` out of the repo. Only use games you legitimately own.

## More

- **Other games**: `nxnative/scripts/build_ipa_from_nsp.py` picks the matching game profile
  under `nxnative/dm/games/<id>/` automatically from the NSP title ID; a game without a
  profile only gets a survey report, no IPA. How to write a profile for a new game:
  `nxnative/dm/games/README.md`.
- **Build options**: `./make_ipa.sh -h` (icon, bundle ID, DLC, diagnostic environment
  switches, …).
- **Upstream sync**: `nxnative/` is a git subtree (not a submodule):

  ```bash
  git fetch <upstream-remote>
  git subtree pull --prefix=nxnative <upstream-remote> main
  ```

- **Running the game on a Mac** (development): see `nxnative/docs/development.md`.
- Port highlights in this fork: `dm/games/0100A00019DE0000/` (game profile and shader
  tables), `dm/tools/patch_nso.py` (the KRES0 widescreen parameter slot),
  `dm/gfx/nvn/nvn_mtl.m` (Metal back-end memory fixes), `dm/ios/Sources/front_ios.m`
  (full screen / widescreen / keep screen awake) and `dm/host/hle_haptics.c` +
  `dm/ios/Sources/Haptics.m` (HD rumble → CoreHaptics).

## License

MIT (see [LICENSE](LICENSE)). Third-party components and their licenses are listed in
`nxnative/README.md` (Acknowledgements) and `nxnative/dm/third_party/README.md`.
