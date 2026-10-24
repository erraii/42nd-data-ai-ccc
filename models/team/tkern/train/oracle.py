"""Exact short-horizon tactics on top of the heuristic: the training teacher.

Works from planes alone (what the referee hands an evaluator), by rebuilding
the board with ``Board.from_observation``. For the side to move it searches
every sequence of its remaining actions this turn: a forced win scores +1,
otherwise the best material it can capture is added to the static score. The
same search for the opponent, as if we passed, estimates what we leave hanging.
"""

from __future__ import annotations

import numpy as np

from colony_one import constants, encoding
from colony_one.board import Board
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team

from heuristic import PIECE_VALUES, heuristic_score

WIN = 10_000.0
_VALUE = {piece_type: float(PIECE_VALUES[piece_type.encode_index()])
          for piece_type in PieceType}


def _opponent(team: str) -> str:
    return constants.MAGENTA if team == constants.CYAN else constants.CYAN


def _turn_end(board: Board, team: str, shrink_now: bool) -> float:
    """Gain for ``team`` resolved at the end of its turn (WIN, -WIN or material)."""
    if board.only_kernels_remain():
        return WIN if board.kernel_endgame_winner(acting_team=team) == team else -WIN
    if not (team == constants.MAGENTA and shrink_now):
        return 0.0
    record = board.apply_firewall_shrink()
    if record is None:
        return 0.0
    try:
        kernels = [p for _, _, p in record.killed if p.piece_type is PieceType.KERNEL]
        if len(kernels) == 2:
            return WIN  # Magenta takes the tiebreak, and only Magenta gets here
        if kernels:
            return WIN if kernels[0].team != team else -WIN
        return sum(_VALUE[p.piece_type] * (1 if p.team != team else -1)
                   for _, _, p in record.killed)
    finally:
        board.undo_firewall_shrink(record)


def best_turn(board: Board, team: str, actions_left: int, shrink_now: bool) -> float:
    """Best gain ``team`` can force with its remaining actions this turn."""
    best = -float("inf")
    for action in legal_actions_for_team(board, team):
        record = board.apply_action(action)
        try:
            captured = record.captured_piece
            if captured is not None and captured.piece_type is PieceType.KERNEL:
                return WIN
            gain = _VALUE[captured.piece_type] if captured is not None else 0.0
            if actions_left > 1:
                gain += best_turn(board, team, actions_left - 1, shrink_now)
            else:
                gain += _turn_end(board, team, shrink_now)
        finally:
            board.undo_action(record)
        if gain >= WIN:
            return WIN
        best = max(best, gain)
    return best if best > -float("inf") else 0.0


def decode_scalars(planes: np.ndarray) -> tuple[str, int, bool]:
    team = constants.MAGENTA if planes[encoding.SIDE_TO_MOVE_PLANE, 0, 0] > 0.5 else constants.CYAN
    actions_left = 2 if planes[encoding.ACTIONS_LEFT_PLANE, 0, 0] > 0.75 else 1
    shrink_now = bool(planes[encoding.URGENCY_PLANE, 0, 0] > 0.999)
    return team, actions_left, shrink_now


def tactical_value(planes: np.ndarray) -> float:
    team, actions_left, shrink_now = decode_scalars(planes)
    board = Board.from_observation(planes, team)
    gain = best_turn(board, team, actions_left, shrink_now)
    if gain >= WIN:
        return 1.0
    # Null move: what could the opponent take next turn if we did nothing? We
    # cannot see next round's shrink from here, so it is left out.
    threat = best_turn(board, _opponent(team), 2, False)
    if threat >= WIN:
        threat = 120.0
    weight = 0.5 if actions_left == 1 else 0.25
    score = heuristic_score(planes) + 0.9 * max(gain, -WIN / 2) - weight * threat
    return float(np.clip(np.tanh(score / 60.0), -0.95, 0.95))


def oracle_evaluator(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([tactical_value(p) for p in planes], dtype=np.float32)
    return np.zeros((len(planes), encoding.NUM_ACTIONS), dtype=np.float32), values
