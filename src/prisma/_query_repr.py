from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._types import PrismaMethod, TransactionId


@dataclass
class QueryRequest:
    """Structured query payload used by the SQLAlchemy engine instead of a GraphQL string."""

    method: PrismaMethod
    model_name: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    root_selection: list[str] | None = None
    tx_id: TransactionId | None = None
