"""My pre-existing poker bot — written long before I'd heard of dev.fun.

It knows NOTHING about the Arena `table` schema. Its whole world is:
  - my two hole cards as plain strings like "Ah", "Kd"
  - the community cards (list of the same)
  - how much it costs to call, and the current pot

It returns one of: "fold" / "check" / "call" / "raise" plus a raise *fraction
of the pot* (because that's how I happened to think about sizing).

This is the thing I want to plug into the sandbox without rewriting.
"""

RANKS = "23456789TJQKA"


def _rank_val(card: str) -> int:
    return RANKS.index(card[0].upper())


def hand_strength(hole, board) -> float:
    """Crude 0..1 strength score. Pure heuristic, no equity calc."""
    if len(hole) != 2:
        return 0.0
    r0, r1 = _rank_val(hole[0]), _rank_val(hole[1])
    suited = hole[0][-1] == hole[1][-1]
    pair = hole[0][0] == hole[1][0]

    score = (r0 + r1) / 24.0          # high-card baseline (0..1)
    if pair:
        score += 0.35 + r0 * 0.01     # pairs are strong, bigger pairs stronger
    if suited:
        score += 0.05
    if abs(r0 - r1) == 1:             # connected
        score += 0.03

    # postflop: reward making a pair with the board
    if board:
        ranks_on_board = {c[0].upper() for c in board}
        if hole[0][0].upper() in ranks_on_board or hole[1][0].upper() in ranks_on_board:
            score += 0.25
        if pair:                      # an overpair / set-ish
            score += 0.15

    return max(0.0, min(score, 1.0))


def decide(hole, board, to_call: int, pot: int):
    """My bot's native decision. Returns (action, raise_pot_fraction|None)."""
    s = hand_strength(hole, board)

    if s >= 0.65:                     # strong -> raise 75% pot
        return ("raise", 0.75)
    if s >= 0.45:                     # decent -> call if cheap, else raise small
        if to_call == 0:
            return ("raise", 0.5)
        return ("call", None)
    # weak
    if to_call == 0:
        return ("check", None)
    # call only if it's basically free relative to pot
    if to_call <= pot * 0.1:
        return ("call", None)
    return ("fold", None)
