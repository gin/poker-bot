"""Tabular CFR trainer for Kuhn poker.

Kuhn poker is intentionally small: three cards, one private card per player,
one betting round, and exact chance enumeration. It is useful here as a
regression-safe CFR foundation before adding larger Hold'em abstractions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import permutations, product
from pathlib import Path

CARDS = ("J", "Q", "K")
CARD_STRENGTH = {card: index for index, card in enumerate(CARDS)}
ACTION_LABELS = {
    "x": "check",
    "b": "bet",
    "c": "call",
    "f": "fold",
}
PLAYER_HISTORIES = {
    0: ("", "xb"),
    1: ("x", "b"),
}


@dataclass
class InfoSetNode:
    key: str
    actions: tuple[str, ...]
    regret_sum: dict[str, float]
    strategy_sum: dict[str, float]

    @classmethod
    def create(cls, key, actions):
        return cls(
            key=key,
            actions=tuple(actions),
            regret_sum={action: 0.0 for action in actions},
            strategy_sum={action: 0.0 for action in actions},
        )

    def strategy(self, realization_weight=0.0):
        positive_regrets = {
            action: max(0.0, self.regret_sum[action]) for action in self.actions
        }
        normalizer = sum(positive_regrets.values())
        if normalizer > 0:
            strategy = {
                action: positive_regrets[action] / normalizer
                for action in self.actions
            }
        else:
            probability = 1.0 / len(self.actions)
            strategy = {action: probability for action in self.actions}

        for action, probability in strategy.items():
            self.strategy_sum[action] += realization_weight * probability
        return strategy

    def average_strategy(self):
        normalizer = sum(self.strategy_sum.values())
        if normalizer <= 0:
            probability = 1.0 / len(self.actions)
            return {action: probability for action in self.actions}
        return {
            action: self.strategy_sum[action] / normalizer for action in self.actions
        }


@dataclass(frozen=True)
class KuhnCfrReport:
    iterations: int
    average_game_value: float
    player0_best_response: float
    player1_best_response: float
    nash_conv: float
    exploitability: float
    strategy: dict[str, dict[str, float]]


def legal_actions(history):
    if history in {"", "x"}:
        return ("x", "b")
    if history in {"b", "xb"}:
        return ("c", "f")
    return ()


def current_player(history):
    if history in {"", "bc", "bf", "xx"}:
        return 0
    if history in {"x", "xbc", "xbf"}:
        return 1
    return len(history) % 2


def terminal_utility_player0(cards, history):
    if history == "xx":
        return 1.0 if CARD_STRENGTH[cards[0]] > CARD_STRENGTH[cards[1]] else -1.0
    if history == "bc":
        return 2.0 if CARD_STRENGTH[cards[0]] > CARD_STRENGTH[cards[1]] else -2.0
    if history == "bf":
        return 1.0
    if history == "xbc":
        return 2.0 if CARD_STRENGTH[cards[0]] > CARD_STRENGTH[cards[1]] else -2.0
    if history == "xbf":
        return -1.0
    return None


def info_set_key(card, history):
    return f"{card}|{history or '-'}"


def action_name(action):
    return ACTION_LABELS[action]


def readable_strategy(strategy):
    return {
        key: {action_name(action): probability for action, probability in row.items()}
        for key, row in sorted(strategy.items())
    }


class KuhnCfrTrainer:
    def __init__(self):
        self.nodes: dict[str, InfoSetNode] = {}
        self.iterations = 0
        self.utility_sum = 0.0

    def node_for(self, card, history):
        key = info_set_key(card, history)
        actions = legal_actions(history)
        if key not in self.nodes:
            self.nodes[key] = InfoSetNode.create(key, actions)
        return self.nodes[key]

    def cfr(self, cards=None, history="", reach0=1.0, reach1=1.0, *, next_history=None):
        if next_history is not None:
            history = next_history
        if cards is None:
            raise ValueError("cards are required")

        terminal = terminal_utility_player0(cards, history)
        if terminal is not None:
            return terminal

        player = current_player(history)
        node = self.node_for(cards[player], history)
        realization_weight = reach0 if player == 0 else reach1
        strategy = node.strategy(realization_weight)
        action_utilities = {}
        node_utility = 0.0

        for action in node.actions:
            next_history = history + action
            if player == 0:
                utility = self.cfr(
                    cards,
                    next_history,
                    reach0 * strategy[action],
                    reach1,
                )
            else:
                utility = self.cfr(
                    cards,
                    next_history,
                    reach0,
                    reach1 * strategy[action],
                )
            action_utilities[action] = utility
            node_utility += strategy[action] * utility

        opponent_reach = reach1 if player == 0 else reach0
        regret_sign = 1.0 if player == 0 else -1.0
        for action in node.actions:
            regret = regret_sign * (action_utilities[action] - node_utility)
            node.regret_sum[action] += opponent_reach * regret
        return node_utility

    def train(self, iterations):
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        deals = tuple(permutations(CARDS, 2))
        for _iteration in range(iterations):
            iteration_utility = 0.0
            for cards in deals:
                iteration_utility += self.cfr(cards)
            self.utility_sum += iteration_utility / len(deals)
            self.iterations += 1
        return self.report()

    def average_strategy(self):
        return {
            key: node.average_strategy() for key, node in sorted(self.nodes.items())
        }

    def report(self):
        strategy = self.average_strategy()
        player0_br = best_response_value(strategy, player=0)
        player1_br = best_response_value(strategy, player=1)
        nash_conv = player0_br + player1_br
        return KuhnCfrReport(
            iterations=self.iterations,
            average_game_value=(
                self.utility_sum / self.iterations if self.iterations else 0.0
            ),
            player0_best_response=player0_br,
            player1_best_response=player1_br,
            nash_conv=nash_conv,
            exploitability=nash_conv / 2.0,
            strategy=readable_strategy(strategy),
        )


def strategy_probability(strategy, card, history, action):
    row = strategy.get(info_set_key(card, history))
    if not row:
        return 1.0 / len(legal_actions(history))
    return row.get(action, 0.0)


def infosets_for_player(player):
    return tuple(
        (info_set_key(card, history), legal_actions(history))
        for card in CARDS
        for history in PLAYER_HISTORIES[player]
    )


def pure_responses_for_player(player):
    infosets = infosets_for_player(player)
    for choices in product(*(actions for _key, actions in infosets)):
        yield {
            key: action
            for (key, _actions), action in zip(infosets, choices, strict=True)
        }


def response_value_for_deal(strategy, response, player, cards, history=""):
    terminal = terminal_utility_player0(cards, history)
    if terminal is not None:
        return terminal if player == 0 else -terminal

    acting = current_player(history)
    actions = legal_actions(history)
    if acting == player:
        action = response[info_set_key(cards[acting], history)]
        return response_value_for_deal(
            strategy,
            response,
            player,
            cards,
            history + action,
        )

    return sum(
        strategy_probability(strategy, cards[acting], history, action)
        * response_value_for_deal(
            strategy,
            response,
            player,
            cards,
            history + action,
        )
        for action in actions
    )


def response_value(strategy, response, player):
    deals = tuple(permutations(CARDS, 2))
    return sum(
        response_value_for_deal(strategy, response, player, cards)
        for cards in deals
    ) / len(deals)


def best_response_value(strategy, player):
    return max(
        response_value(strategy, response, player)
        for response in pure_responses_for_player(player)
    )


def train_kuhn(iterations):
    trainer = KuhnCfrTrainer()
    return trainer.train(iterations)


def report_to_jsonable(report):
    return asdict(report)


def write_json_report(report, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report_to_jsonable(report), f, indent=2)
        f.write("\n")


def format_report(report):
    lines = [
        "# Kuhn CFR",
        "",
        f"iterations: {report.iterations}",
        f"average_game_value_p0: {report.average_game_value:+.4f}",
        f"player0_best_response: {report.player0_best_response:+.4f}",
        f"player1_best_response: {report.player1_best_response:+.4f}",
        f"nash_conv: {report.nash_conv:+.4f}",
        f"exploitability: {report.exploitability:+.4f}",
        "",
        "## Average Strategy",
        "",
        "| infoset | actions |",
        "| --- | --- |",
    ]
    for key, actions in sorted(report.strategy.items()):
        action_text = ", ".join(
            f"{action} {probability:.1%}" for action, probability in actions.items()
        )
        lines.append(f"| `{key}` | {action_text} |")
    return "\n".join(lines)


def write_markdown_report(report, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_report(report))
