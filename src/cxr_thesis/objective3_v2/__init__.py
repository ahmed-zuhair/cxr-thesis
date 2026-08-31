"""Objective 3 v2.0: quantum enhancement, redesigned.

A new preregistered study motivated by the v1.1 null result. v1.0 and v1.1 stay
published and final; nothing here amends them retrospectively.

The v1.1 quantum layer acted on a 160-dimensional vector produced by a frozen
classical GAT, so the graph structure was already collapsed before the circuit
saw anything. This study replaces that with an entangling pattern determined by
the graph adjacency, sizes the comparison from a power analysis rather than a
three-seed win count, and predicts the achievable advantage from the data itself
before training.
"""

from __future__ import annotations

STUDY = "objective3_v2"
VERSION = "v2.0.0"

__all__ = ["STUDY", "VERSION"]
