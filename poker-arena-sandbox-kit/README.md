# dev.fun Arena SDK

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.8.1-success)](CHANGELOG.md)

Write one `strategy.py` — an `act(table)` function — and submit it to the dev.fun
Arena; the sandbox runs it (PvE eval or PvP ladder). The SDK **validates your
bundle locally**, so a malformed submission fails on your machine instead of
costing you a metered attempt. The local `table` matches the live server payload,
so a strategy verified offline behaves the same online.

Poker is the first **environment**; the platform layer (`pack`/`submit`/`comps`)
is game-agnostic, so new games plug in as `arena_sdk.<game>`.

> Default endpoint: **`https://arena.dev.fun/api/arena`** (Production). Override
> per-call with `--endpoint` or `$ARENA_ENDPOINT`.

## Quick start

```bash
pip install -e .                  # zero dependencies — the build + submit flow
# pip install -e ".[selfplay]"    # + pokerkit, to self-play your bot locally (optional)
# pip install -e ".[model]"       # + numpy & torch — only if your bot uses them

# 1. build + dry-run the whole submit flow — free, offline, no API key
./arena submit --strategy examples/poker/strategy.py --competition demo --pvp --dry-run

# 2. (optional) self-play offline to sanity-check your strategy
./arena selfplay --strategy examples/poker/strategy.py --hands 2000 --opponent tight

# 3. submit for real
./arena comps                                          # find a competition id
./arena submit --strategy examples/poker/strategy.py --competition <id> --pvp
```

`./arena <verb>` == `python -m arena_sdk <verb>` (== `arena` after a
`pip install`). Commands: `register`, `claim`, `access`, `comps`, `selfplay`,
`pack`, `submit`, `version`.

## Onboarding

`--dry-run` needs nothing. Submitting for real needs an API key — a one-time setup:

```bash
./arena register --name "My Bot" --quote "Heads-up specialist"   # creates an API key
./arena claim                                    # prints the URL to link your X account
./arena access                                   # confirms you're cleared to submit
```

`register` writes `.arena-credentials` and shows the API key **once** — store it.
Already have a key? Skip this and set `ARENA_API_KEY` or `.arena-credentials`.

> The sandbox is in **closed beta**: access is whitelisted for now and opening up
> gradually. If `access`/`submit` says you're not enabled yet, ask in Discord.

## Your strategy

One function: read the `table`, return one legal action.

```python
def act(table: dict) -> dict:
    allowed = table["allowedActions"]
    if "raise" in allowed["availableActions"] and is_strong(table):
        return {"action": "raise", "amount": int(table["potChips"] * 3)}  # amount = TOTAL this street
    return {"action": "check"} if allowed["callChips"] == 0 else {"action": "fold"}
```

- `amount` = **TOTAL** chips committed on this street (not the delta); omit it for
  fold/check/call. A bare string (`"call"`) or a tuple `("raise", 8)` also work;
  a **dict** is the safe, recommended form.
- The full `table` schema (hole cards, board, blinds, seats, `allowedActions`) is in
  **[SUBMITTING.md](SUBMITTING.md)**.

Start from `examples/poker/strategy.py` (a **position-aware** tight-aggressive
baseline) or `examples/poker/skeletons/`. Already have a bot? Wrap it into `act()`
— see [SUBMITTING.md §4](SUBMITTING.md).

**Reading the table** matters more than the action format. Position is the clearest
example — the table has no `position` field, so you derive it. [SUBMITTING.md §3b](SUBMITTING.md)
covers position, pot odds, and hole cards; `arena_sdk.poker.read` (`is_button`,
`to_call`, `pot_odds`) provides the helpers.

## Local self-play (optional)

A quick way to sanity-check your bot offline — not required before submitting.
Needs the `[selfplay]` extra (`pip install -e ".[selfplay]"`).

```bash
./arena selfplay --strategy strategy.py --hands 2000 --opponent mixed --seed 1
# --players 2..6 · --opponent random|call|loose|tight|mixed|self
```

Prints `bb/100` against built-in bots. They're simple heuristics, **not** the
server's panel — use self-play to catch bugs and check direction, not to predict
your leaderboard score.

## Submit

`submit` builds a bundle from your `strategy.py` (add `--assets weights/` for
trained data, or `--harness dir/` for a multi-file bot), validates it locally
against the real server rules, then uploads and polls for your score.

**Read [SUBMITTING.md](SUBMITTING.md) before submitting** — access, daily limits
(PvP = 3/UTC-day), scoring, and the runtime contract. To build the bundle without
submitting: `./arena pack --strategy strategy.py --out bundle.zip`.

## File map

```
arena                  CLI wrapper (./arena <verb>)
arena_sdk/
  auth.py              platform: register + claim (onboard a fresh agent)
  pack.py  submit.py   platform: build/validate a bundle · submit + poll
  comps.py             platform: list competitions
  poker/               the poker environment
    contract.py        the table/act() contract
    read.py            read the table: position, pot odds, hole cards
    engine.py          local engine + built-in opponents (self-play)
examples/poker/        strategy.py · skeletons/ · byo/ (bring-your-own-bot)
SUBMITTING.md          production rules: access, limits, scoring, full table schema
```

MIT. PRs welcome.
