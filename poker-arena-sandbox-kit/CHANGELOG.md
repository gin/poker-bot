# Changelog

## 0.8.1

Initial public release.

- Write one `strategy.py` (an `act(table)` function); validate the bundle locally,
  then submit to the dev.fun Arena heads-up sandbox PvP ladder.
- **Zero-dependency core** — `register` / `claim` / `access` / `comps` / `pack` /
  `submit` are pure standard library. Optional extras: `[selfplay]` (pokerkit) for
  local self-play, `[model]` (numpy + torch) for model bots.
- The local `table` mirrors the live server contract: position is derived from
  `recentEvents`, and the return format is validated before a metered submission.
- Verified against Production (`arena.dev.fun`): submission contract, access gates,
  polling, and TrueSkill scoring.
