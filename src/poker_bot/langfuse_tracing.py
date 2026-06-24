"""Langfuse tracing setup and helpers for the poker bot.

Environment variables (all optional — tracing is disabled when unset):
    LANGFUSE_PUBLIC_KEY   – Langfuse public key (pk-lf-...)
    LANGFUSE_SECRET_KEY   – Langfuse secret key (sk-lf-...)
    LANGFUSE_HOST         – Langfuse base URL
                          (default: https://cloud.langfuse.com)
    LANGFUSE_TRACE_POKER  – Set to 0 to disable tracing
                          (default: enabled when keys are set)

Usage:
    from poker_bot.langfuse_tracing import (
        flush_langfuse,
        trace_hand,
        span_decision,
        span_table_process,
        span_strategy_call,
        span_action_result,
    )
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from langfuse import get_client as _get_langfuse_client
from langfuse import propagate_attributes as _propagate


def langfuse_enabled() -> bool:
    """Return True when Langfuse credentials are configured and tracing on."""
    disabled = {"0", "false", "no", "off"}
    if os.environ.get("LANGFUSE_TRACE_POKER", "").lower() in disabled:
        return False
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(pk and sk)


def get_langfuse_client():
    """Return the global Langfuse client, or None if tracing is disabled."""
    if not langfuse_enabled():
        return None
    return _get_langfuse_client()


def flush_langfuse() -> None:
    """Flush any pending Langfuse observations (call before shutdown)."""
    client = get_langfuse_client()
    if client is not None:
        client.flush()


@contextmanager
def trace_hand(
    hand_id: str,
    table_id: str,
    street: str,
    strategy_name: str,
    agent_id: str,
    competition_id: str,
    pot_chips: int,
    num_seats: int,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Generator[Any]:
    """Create a top-level trace for a single poker hand.

    Wraps all processing for one hand at one table. Child spans (decision,
    table-process) nest under this trace.

    Usage:
        with trace_hand(hand_id=..., table_id=..., ...) as trace:
            trace.update(output={"result": "won", "net_chips": 100})
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    metadata: dict[str, Any] = {
        "table_id": table_id,
        "street": street,
        "strategy": strategy_name,
        "agent_id": agent_id,
        "competition_id": competition_id,
        "num_seats": num_seats,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    tags = ["poker", "hand", f"street:{street.lower()}", f"strategy:{strategy_name}"]
    if num_seats <= 2:
        tags.append("heads-up")
    elif num_seats <= 4:
        tags.append("short-handed")
    else:
        tags.append("full-ring")

    with (
        _propagate(
            user_id=user_id or agent_id,
            session_id=session_id or hand_id,
            tags=tags,
            metadata=metadata,
        ),
        client.start_as_current_observation(
            as_type="span",
            name=f"hand-{hand_id}",
            input={
                "hand_id": hand_id,
                "table_id": table_id,
                "street": street,
                "pot_chips": pot_chips,
                "num_seats": num_seats,
            },
        ) as trace,
    ):
        yield trace


@contextmanager
def span_table_process(
    table_id: str,
    street: str,
    pot_chips: int,
    num_opponents: int,
) -> Generator[Any]:
    """Create a span for processing a single table snapshot.

    Nests under the parent hand trace. Captures table-level context.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    with client.start_as_current_observation(
        as_type="span",
        name=f"table-{table_id}",
        input={
            "table_id": table_id,
            "street": street,
            "pot_chips": pot_chips,
            "num_opponents": num_opponents,
        },
        metadata={
            "table_id": table_id,
            "street": street,
        },
    ) as span:
        yield span


@contextmanager
def span_decision(
    decision_index: int,
    street: str,
    position: str,
    hand_description: str,
    facing_bet: bool,
    pot_chips: int,
    stack_chips: int,
    big_blind: int,
) -> Generator[Any]:
    """Create a span for a single poker decision (the strategy call).

    Nests under the table-process span. Captures the full decision context
    so you can see in Langfuse exactly what information the strategy saw.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    with client.start_as_current_observation(
        as_type="span",
        name=f"decision-{decision_index}",
        input={
            "decision_index": decision_index,
            "street": street,
            "position": position,
            "hand": hand_description,
            "facing_bet": facing_bet,
            "pot_chips": pot_chips,
            "stack_chips": stack_chips,
            "big_blind": big_blind,
        },
        metadata={
            "decision_index": decision_index,
            "street": street,
            "position": position,
            "facing_bet": facing_bet,
        },
    ) as span:
        yield span


@contextmanager
def span_strategy_call(strategy_name: str) -> Generator[Any]:
    """Create a generation-type span for the actual strategy invocation.

    This is the innermost span — it wraps the `choose_action()` call and
    captures the action, amount, and reasoning message as output.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    with client.start_as_current_observation(
        as_type="generation",
        name=f"strategy-{strategy_name}",
        metadata={
            "strategy": strategy_name,
        },
    ) as span:
        yield span


@contextmanager
def span_action_result(
    action: str,
    amount: int | None,
    accepted: bool,
    error: str | None = None,
) -> Generator[Any]:
    """Create a span for the arena action submission result."""
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    input_data: dict[str, Any] = {"action": action, "amount": amount}
    output_data: dict[str, Any] = {"accepted": accepted}
    if error:
        output_data["error"] = error

    with client.start_as_current_observation(
        as_type="span",
        name=f"action-{action}",
        input=input_data,
        output=output_data,
        metadata={
            "action": action,
            "accepted": accepted,
            "error": error,
        },
    ) as span:
        yield span
