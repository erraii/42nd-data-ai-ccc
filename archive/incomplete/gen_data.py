"""Generate (planes, value, policy) training data with the oracle under referee MCTS.

    python train/gen_data.py --games 1000 --out train/data4

Writes shards ``<out>_NNN.npz`` every ``--shard`` games. Planes are stored as
uint8 (scalar planes times 8; urgency is a multiple of 1/8) to save memory.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from colony_one import constants, encoding, mcts
from colony_one.baselines import GreedyAgent
from colony_one.env import ColonyOneEnv

from oracle import oracle_evaluator

SIMULATIONS = 100
KEEP_LEAVES = 0.4
DEFAULT_OPPONENTS = [str(Path(__file__).resolve().parents[2] / "model_fixed.onnx")]


def to_uint8(planes: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(planes) * 8).astype(np.uint8)


def from_uint8(planes: np.ndarray) -> np.ndarray:
    out = np.empty(planes.shape, dtype=np.float32)
    out[:, :13] = planes[:, :13] > 0
    out[:, 13:] = planes[:, 13:] / 8.0
    return out


class RecordingEvaluator:
    """Wraps an evaluator and keeps a random share of the positions it labels."""

    def __init__(self, evaluator, rng: np.random.Generator) -> None:
        self.evaluator = evaluator
        self.rng = rng
        self.planes: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    def __call__(self, planes):
        logits, values = self.evaluator(planes)
        keep = self.rng.random(len(planes)) < KEEP_LEAVES
        if keep.any():
            self.planes.append(to_uint8(planes[keep]))
            self.values.append(np.asarray(values)[keep])
        return logits, values


class OracleSearch:
    def __init__(self, evaluator, rng, seed: int) -> None:
        self.evaluator = evaluator
        self.rng = rng
        self.extended = seed % 2 == 0
        self.config = mcts.MCTSConfig(simulations=SIMULATIONS)
        self.roots: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []

    def select(self, env: ColonyOneEnv):
        root = mcts.run(env.board, env.current_team, env.actions_left,
                        env.round_number, self.evaluator, self.config, rng=self.rng)
        self.roots.append(to_uint8(env.observation()[None])[0])
        self.policies.append(mcts.policy_target(root))
        move = mcts.move_number(env.current_team, env.actions_left, env.round_number)
        # Half the games follow the referee's opening schedule; the rest sample
        # for longer, to see more varied middlegames.
        if self.extended:
            temperature = 0.7 if move <= 16 else 0.0
        else:
            temperature = mcts.warmup_temperature(move)
        index = mcts.select_action_index(root, temperature, rng=self.rng)
        return encoding.decode_action(index, env.current_team, env.board)


class BaselineAgent:
    def __init__(self, agent) -> None:
        self.agent = agent

    def select(self, env: ColonyOneEnv):
        return self.agent.select(env.board, env.current_team, env.actions_left,
                                 env.round_number)


class ModelSearch(BaselineAgent):
    def __init__(self, path: Path, rng) -> None:
        from colony_one.onnx_runner import OnnxEvaluator
        self.evaluator = OnnxEvaluator(path)
        self.rng = rng
        self.config = mcts.MCTSConfig(simulations=SIMULATIONS)

    def select(self, env: ColonyOneEnv):
        return mcts.search_action(env.board, env.current_team, env.actions_left,
                                  env.round_number, self.evaluator, self.config,
                                  rng=self.rng)


def play_game(job):
    seed, opponents = job
    rng = np.random.default_rng(seed)
    recorder = RecordingEvaluator(oracle_evaluator, rng)
    ours = OracleSearch(recorder, rng, seed)
    mode = seed % 5
    our_team = constants.CYAN if (seed // 5) % 2 == 0 else constants.MAGENTA
    if mode in (0, 1):
        other = ours  # self-play: both sides record
    elif mode in (2, 3):
        other = BaselineAgent(GreedyAgent(seed + 7))
    else:
        other = ModelSearch(Path(opponents[(seed // 10) % len(opponents)]), rng)

    env = ColonyOneEnv()
    while not env.terminated:
        agent = ours if env.current_team == our_team else other
        env.step(agent.select(env))

    planes = np.concatenate(recorder.planes) if recorder.planes else np.zeros((0, 16, 8, 8), np.uint8)
    values = np.concatenate(recorder.values) if recorder.values else np.zeros(0, np.float32)
    _, unique = np.unique(planes.reshape(len(planes), -1), axis=0, return_index=True)
    return (planes[unique], values[unique], np.stack(ours.roots),
            np.stack(ours.policies).astype(np.float16), env.winner == our_team, mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=100)
    parser.add_argument("--out", type=str, default="train/data4")
    parser.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS,
                        help="ONNX models the oracle plays in mode 4")
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="which game modes to play (seed %% 5)")
    args = parser.parse_args()

    wins = {m: [0, 0] for m in range(5)}
    buffer: list[tuple] = []
    shard = 0

    def flush() -> None:
        nonlocal buffer, shard
        if not buffer:
            return
        path = Path(f"{args.out}_{args.seed:06d}_{shard:03d}.npz")
        np.savez_compressed(
            path,
            planes=np.concatenate([b[0] for b in buffer]),
            values=np.concatenate([b[1] for b in buffer]).astype(np.float32),
            roots=np.concatenate([b[2] for b in buffer]),
            policies=np.concatenate([b[3] for b in buffer]),
        )
        print(f"wrote {path} ({sum(len(b[0]) for b in buffer)} positions)", flush=True)
        buffer, shard = [], shard + 1

    seeds = [s for s in range(args.seed, args.seed + 5 * args.games) if s % 5 in args.modes]
    jobs = [(seed, args.opponents) for seed in seeds[:args.games]]
    with Pool() as pool:
        for i, result in enumerate(pool.imap_unordered(play_game, jobs, chunksize=1)):
            buffer.append(result[:4])
            wins[result[5]][0] += result[4]
            wins[result[5]][1] += 1
            if (i + 1) % args.shard == 0:
                flush()
    flush()
    names = ["self-play", "self-play", "vs greedy", "vs greedy", "vs partner"]
    for mode, (w, n) in wins.items():
        print(f"  mode {mode} ({names[mode]}): oracle side won {w}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
