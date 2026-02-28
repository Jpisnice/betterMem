from __future__ import annotations

from .api.client import BetterMem

__all__ = ["BetterMem"]


def main() -> None:
    """Entry point for the `bettermem` console script."""
    print("BetterMem package is installed. Import `BetterMem` from `bettermem` to use the API.")
