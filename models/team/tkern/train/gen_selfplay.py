"""Self-play data from a trained network: the AlphaZero improvement step.

    python train/gen_selfplay.py --model models/v1.onnx --games 1000 --out train/sp1 \
        --opponents "../level4 current model/model.onnx" ../level4update/selfplay/best.onnx

The network searches with more simulations than the referee, so its visit
counts are a stronger policy than its raw output. Every searched root becomes
a training position: policy = visit distribution, value = a blend of the game
result and the search's own root value. Shards use the same layout as
``gen_data.py``, so ``train_net.py`` reads both.
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
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator

from gen_data import to_uint8

RESULT_WEIGHT = 0.5
SELF_PLAY_SHARE = 0.6


class Searcher:
    def __init__(self, path: Path, simulations: int, rng, record: bool) -> None:
        self.evaluator = OnnxEvaluator(path)
        self.config = mcts.MCTSConfig(simulations=simulations)
        self.rng = rng
        self.record = record

    def select(self, env: ColonyOneEnv, temperature: float, log: list) -> int:
        root = mcts.run(env.board, env.current_team, env.actions_left,
                        env.round_number, self.evaluator, self.config, rng=self.rng)
        if self.record:
            visits = sum(root.N.values())
            q = sum(root.W.values()) / visits if visits else 0.0
            log.append((to_uint8(env.observation()[None])[0], mcts.policy_target(root),
                        q, env.current_team))
        return mcts.select_action_index(root, temperature, rng=self.rng)


def play_game(job):
    seed, model, opponents, simulations = job
    rng = np.random.default_rng(seed)
    ours = Searcher(Path(model), simulations, rng, record=True)
    if rng.random() < SELF_PLAY_SHARE or not opponents:
        other = ours
    else:
        # The opponent searches at the referee's budget and is not recorded.
        other = Searcher(Path(opponents[seed % len(opponents)]), 100, rng, record=False)
    our_team = constants.CYAN if seed % 2 == 0 else constants.MAGENTA
    sample_until = 16 if seed % 4 < 2 else 8

    env = ColonyOneEnv()
    log: list = []
    while not env.terminated:
        agent = ours if env.current_team == our_team else other
        move = mcts.move_number(env.current_team, env.actions_left, env.round_number)
        temperature = (1.0 if move <= 4 else 0.6) if move <= sample_until else 0.0
        index = agent.select(env, temperature, log)
        env.step(encoding.decode_action(index, env.current_team, env.board))

    planes = np.stack([p for p, _, _, _ in log])
    policies = np.stack([pol for _, pol, _, _ in log]).astype(np.float16)
    z = np.array([1.0 if team == env.winner else -1.0 for _, _, _, team in log])
    q = np.array([q for _, _, q, _ in log])
    values = (RESULT_WEIGHT * z + (1 - RESULT_WEIGHT) * q).astype(np.float32)
    return planes, values, policies, other is ours, env.winner == our_team


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=250)
    parser.add_argument("--out", type=str, default="train/sp")
    args = parser.parse_args()

    jobs = [(seed, args.model, args.opponents, args.simulations)
            for seed in range(args.seed, args.seed + args.games)]
    buffer: list[tuple] = []
    shard = 0
    vs_wins = vs_games = 0

    def flush() -> None:
        nonlocal buffer, shard
        if not buffer:
            return
        planes = np.concatenate([b[0] for b in buffer])
        path = Path(f"{args.out}_{args.seed:06d}_{shard:03d}.npz")
        np.savez_compressed(
            path, planes=planes,
            values=np.concatenate([b[1] for b in buffer]),
            roots=planes, policies=np.concatenate([b[2] for b in buffer]),
        )
        print(f"wrote {path} ({len(planes)} positions)", flush=True)
        buffer, shard = [], shard + 1

    with Pool() as pool:
        for i, result in enumerate(pool.imap_unordered(play_game, jobs, chunksize=1)):
            buffer.append(result[:3])
            if not result[3]:
                vs_games += 1
                vs_wins += result[4]
            if (i + 1) % args.shard == 0:
                flush()
    flush()
    if vs_games:
        print(f"vs opponents at 200 vs 100 sims: won {vs_wins}/{vs_games}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
