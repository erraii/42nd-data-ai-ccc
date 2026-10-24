"""A hand-written planes-only evaluator, used as a teacher and a yardstick."""

from __future__ import annotations

import numpy as np

# Plane order: Ping, Daemon, Exploit, Trace, Ice, Kernel (own 0-5, enemy 6-11).
PIECE_VALUES = np.array([15, 45, 25, 30, 50, 0], dtype=np.float32)

_rows, _cols = np.mgrid[0:8, 0:8]
# Chebyshev ring index: 0 = outer ring, 3 = centre 2x2.
RING = np.minimum(np.minimum(_rows, 7 - _rows), np.minimum(_cols, 7 - _cols)).astype(np.float32)


def _kernel_score(kernel: np.ndarray, guards: np.ndarray, urgency: np.ndarray) -> np.ndarray:
    ring = (kernel * RING).sum(axis=(1, 2))
    # Count friendly non-Kernel pieces adjacent to the Kernel.
    padded = np.pad(guards, ((0, 0), (1, 1), (1, 1)))
    neigh = sum(padded[:, 1 + dr:9 + dr, 1 + dc:9 + dc]
                for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    guarded = (kernel * neigh).sum(axis=(1, 2))
    return ring * (2.0 + 6.0 * urgency) + 6.0 * np.minimum(guarded, 1.0)


def heuristic_scores(planes: np.ndarray) -> np.ndarray:
    """Static score in material points, from the side to move's view."""
    planes = np.asarray(planes, dtype=np.float32)
    own = planes[:, 0:6].sum(axis=(2, 3)) @ PIECE_VALUES
    enemy = planes[:, 6:12].sum(axis=(2, 3)) @ PIECE_VALUES
    urgency = planes[:, 14, 0, 0]
    own_k = _kernel_score(planes[:, 5], planes[:, 0:5].sum(axis=1), urgency)
    enemy_k = _kernel_score(planes[:, 11], planes[:, 6:11].sum(axis=1), urgency)
    return (own - enemy) + (own_k - enemy_k)


def heuristic_score(planes: np.ndarray) -> float:
    return float(heuristic_scores(planes[None])[0])


def heuristic_evaluator(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.tanh(heuristic_scores(planes) / 40.0).astype(np.float32)
    return np.zeros((len(planes), 4096), dtype=np.float32), value
