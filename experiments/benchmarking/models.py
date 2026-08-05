"""Model registry: which models to benchmark.

Each call to ``get_models()`` returns fresh, uninitialised instances
so they can be independently trained or loaded without shared state.
"""

from __future__ import annotations

from experiments.models import DeepLabV3RN50


def get_models() -> list:
    """Return fresh model instances for benchmarking."""
    return [
        DeepLabV3RN50(),
    ]
