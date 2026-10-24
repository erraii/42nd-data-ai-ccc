"""Head-to-head matches between evaluators (under referee MCTS) and baselines.

    python train/arena.py model_a.onnx model_b.onnx --games 40
    python train/arena.py model.onnx greedy --games 40

An agent is an ONNX path, ``greedy``, ``random``, ``heuristic`` or ``oracle``.
Sides alternate every game; games run in parallel.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from colony_one import constants, mcts
from colony_one.baselines import GreedyAgent, RandomAgent
from colony_one.env import ColonyOneEnv


class SearchAgent:
    def __init__(self, evaluator, simulations: int, seed: int) -> None:
        self.evaluator = evaluator
        self.config = mcts.MCTSConfig(simulations=simulations)
        self.rng = np.random.default_rng(seed)

    def select(self, board, team, actions_left, round_number):
        return mcts.search_action(board, team, actions_left, round_number,
                                  self.evaluator, self.config, rng=self.rng)


def make_evaluator(spec: str):
    if spec == "heuristic":
        from heuristic import heuristic_evaluator
        return heuristic_evaluator
    if spec == "oracle":
        from oracle import oracle_evaluator
        return oracle_evaluator
    from colony_one.onnx_runner import OnnxEvaluator
    return OnnxEvaluator(Path(spec))


def make_agent(spec: str, simulations: int, seed: int):
    if spec == "greedy":
        return GreedyAgent(seed=seed)
    if spec == "random":
        return RandomAgent(seed=seed)
    return SearchAgent(make_evaluator(spec), simulations, seed)


def play_game(job):
    spec_a, spec_b, game, simulations = job
    agent_a = make_agent(spec_a, simulations, 2 * game)
    agent_b = make_agent(spec_b, simulations, 2 * game + 1)
    a_team = constants.CYAN if game % 2 == 0 else constants.MAGENTA
    env = ColonyOneEnv()
    while not env.terminated:
        agent = agent_a if env.current_team == a_team else agent_b
        env.step(agent.select(env.board, env.current_team, env.actions_left,
                              env.round_number))
    return a_team, env.winner == a_team, env.action


def match(spec_a: str, spec_b: str, games: int, simulations: int = 100, seed: int = 0):
    jobs = [(spec_a, spec_b, seed + g, simulations) for g in range(games)]
    with Pool() as pool:
        results = pool.map(play_game, jobs, chunksize=1)
    by_side = {constants.CYAN: [0, 0], constants.MAGENTA: [0, 0]}
    for team, won, _ in results:
        by_side[team][0] += won
        by_side[team][1] += 1
    wins = sum(won for _, won, _ in results)
    length = float(np.mean([n for _, _, n in results]))
    return wins / games, by_side, length


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rate, by_side, length = match(args.a, args.b, args.games, args.simulations, args.seed)
    c, m = by_side[constants.CYAN], by_side[constants.MAGENTA]
    print(f"{Path(args.a).name} vs {Path(args.b).name}: A wins {rate:.0%} "
          f"(as CYAN {c[0]}/{c[1]}, as MAGENTA {m[0]}/{m[1]}), "
          f"avg {length:.0f} actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
