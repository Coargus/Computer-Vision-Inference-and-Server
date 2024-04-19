"""Status Enum for the CV API."""

from __future__ import annotations

import enum


class Status(enum.Enum):
    """Status Enum for the CV API."""

    UNKNOWN = 0
    SUCCESS = 1
    RUNNING = 2
    FAILURE = 3
    INVALID = 4
