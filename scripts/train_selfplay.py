#!/usr/bin/env python3
"""Iterative CPU self-play for Colony: One Level 4. Standalone entry point.
Run in the official environment with numpy, torch, onnx and onnxruntime.
Matches are a compute budget, not a guarantee of playing strength.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import hashlib
import json
import math
import multiprocessing
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from colony_one import ColonyOneEnv, constants, encoding, mcts
from colony_one.baselines import GreedyAgent

def teacher(observations):
    """Fast heuristic used only to generate training examples."""
    flat = observations.reshape(-1, 16, 64)
    own, enemy = flat[:, :6], flat[:, 6:12]
    weights = np.array([1.0, 2.5, 1.8, 2.0, 2.8, 0.0], np.float32)
    material = ((own - enemy).sum(axis=2) * weights).sum(axis=1)
    squares = np.arange(64)
    rows, cols = squares // 8, squares % 8
    centre = (3.5 - np.maximum(abs(rows - 3.5), abs(cols - 3.5))) / 3.0
    centre = centre.astype(np.float32)
    king_centre = ((own[:, 5] - enemy[:, 5]) * centre).sum(axis=1)
    adjacent = (
        np.maximum(abs(rows[:, None] - rows), abs(cols[:, None] - cols)) == 1
    ).astype(np.float32)
    own_guards = ((own[:, :5].sum(axis=1) @ adjacent) * own[:, 5]).sum(axis=1)
    enemy_guards = ((enemy[:, :5].sum(axis=1) @ adjacent) * enemy[:, 5]).sum(axis=1)
    # Next ring is the open perimeter adjacent to an edge or locked square.
    locked = observations[:, 12]
    rim = np.zeros_like(locked)
    rim[:, 0, :] = rim[:, -1, :] = 1
    rim[:, :, 0] = rim[:, :, -1] = 1
    rim[:, 1:, :] = np.maximum(rim[:, 1:, :], locked[:, :-1, :])
    rim[:, :-1, :] = np.maximum(rim[:, :-1, :], locked[:, 1:, :])
    rim[:, :, 1:] = np.maximum(rim[:, :, 1:], locked[:, :, :-1])
    rim[:, :, :-1] = np.maximum(rim[:, :, :-1], locked[:, :, 1:])
    rim = (rim * (1 - locked)).reshape(-1, 64)
    urgency = flat[:, 14, 0]
    burning = ((enemy[:, 5] - own[:, 5]) * rim).sum(axis=1)
    value = np.tanh(
        0.45 * material + 0.45 * king_centre
        + 0.12 * (own_guards - enemy_guards)
        + 1.2 * urgency ** 4 * burning
    ).astype(np.float32)
    # Gentle priors; MCTS handles legal moves, bodyguards and tactics exactly.
    capture = (enemy * np.array([0.6, 0.9, 0.8, 0.8, 1, 1.2], np.float32)[None, :, None]).sum(axis=1)
    delta = centre[None, None, :] - centre[None, :, None]
    king_from = own[:, 5, :, None]
    policy = (
        capture[:, None, :] + (0.15 + 0.5 * king_from) * delta
        - 0.7 * king_from * rim[:, None, :] * urgency[:, None, None] ** 4
    ).reshape(-1, 4096).astype(np.float32)
    return policy, value

def make_model():
    import torch
    from torch import nn

    class PolicyValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Flatten(1), nn.Linear(1024, 64), nn.ReLU(),
            )
            self.policy = nn.Linear(64, 4096)
            self.value = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

        def forward(self, observation):
            hidden = self.body(observation)
            # The official validator uses exact float32 bounds. Leave a small
            # margin even when an inference kernel rounds saturated Tanh.
            value = torch.clamp(self.value(hidden).squeeze(1), -0.999999, 0.999999)
            return self.policy(hidden), value

    return PolicyValue()

def export(model, destination):
    import onnx
    import onnxruntime as ort
    import torch

    # Legacy exporter uses constant Reshape/Squeeze arguments at opset 17.
    torch.onnx.export(
        model, torch.zeros(1, 16, 8, 8), str(destination),
        input_names=['observation'], output_names=['policy', 'value'],
        dynamic_axes={name: {0: 'batch'} for name in ['observation', 'policy', 'value']},
        opset_version=17, dynamo=False, external_data=False,
    )
    graph = onnx.load(str(destination))
    # The bundled checker rejects Constant nodes, although its documentation
    # describes constant shape arguments. Store those tensors as initializers
    # instead. This preserves their names, values and graph semantics.
    remaining_nodes = []
    for node in graph.graph.node:
        if node.op_type != 'Constant':
            remaining_nodes.append(node)
            continue
        attributes = {a.name: a for a in node.attribute}
        if len(node.output) != 1 or 'value' not in attributes:
            raise RuntimeError('Unsupported Constant encoding; expected a tensor value.')
        tensor = onnx.TensorProto()
        tensor.CopyFrom(attributes['value'].t)
        tensor.name = node.output[0]
        graph.graph.initializer.append(tensor)
    del graph.graph.node[:]
    graph.graph.node.extend(remaining_nodes)
    onnx.checker.check_model(graph)
    onnx.save_model(graph, str(destination), save_as_external_data=False)
    from colony_one.onnx_runner import check_operators
    check_operators(destination)
    if destination.stat().st_size > 5 * 1024 ** 2:
        raise RuntimeError('ONNX model exceeds the 5 MiB limit')
    # Check dynamic batches AND compare actual game-board inference to PyTorch.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=['CPUExecutionProvider'])
    obs = ColonyOneEnv().observation()
    for batch in [1, 8, 16]:
        boards = np.repeat(obs[None], batch, axis=0)
        actual = session.run(None, {'observation': boards})
        with torch.inference_mode():
            expected = model(torch.from_numpy(boards))
        assert actual[0].shape == (batch, 4096) and actual[1].shape == (batch,)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e.numpy(), rtol=1e-4, atol=1e-5)
        assert np.isfinite(actual[0]).all() and np.isfinite(actual[1]).all()
        assert (abs(actual[1]) <= 1.0).all()
    print(f"Export OK: {destination} ({destination.stat().st_size/1024**2:.2f} MiB); batches 1/8/16 checked", flush=True)


_CURRENT = None
_RIVALS = []
_ARENA_CANDIDATE = None
_ARENA_CHAMPION = None


def worker_init(current_path, rival_paths):
    global _CURRENT, _RIVALS
    from colony_one.onnx_runner import OnnxEvaluator
    _CURRENT = OnnxEvaluator(current_path) if current_path else teacher
    _RIVALS = [OnnxEvaluator(p) for p in rival_paths]


def training_game(job):
    game_id, seed, simulations = job
    rng = np.random.default_rng(seed)
    env = ColonyOneEnv(seed=seed)
    config = mcts.MCTSConfig(
        simulations=simulations, warmup_temperatures=(1.0, 0.75, 0.5, 0.25),
    )
    learner_team = constants.CYAN if (game_id // 10) % 2 == 0 else constants.MAGENTA
    mode = game_id % 10
    # 70% latest self-play, 10% recent accepted model, 10% heuristic, 10% greedy.
    rival = _CURRENT
    if mode == 7 and _RIVALS:
        rival = _RIVALS[int(rng.integers(len(_RIVALS)))]
    elif mode == 8:
        rival = teacher
    greedy = GreedyAgent(seed=seed + 1)
    samples = []
    while not env.terminated:
        team = env.current_team
        if mode == 9 and team != learner_team:
            action = greedy.select(env.board, team, env.actions_left, env.round_number)
        else:
            is_learner = team == learner_team or mode < 7
            evaluator = _CURRENT if is_learner else rival
            root = mcts.run(env.board, team, env.actions_left,
                            env.round_number, evaluator, config, rng=rng)
            if is_learner:
                indices = env.legal_action_indices().astype(np.int16)
                probabilities = mcts.policy_target(root)[indices]
                if probabilities.sum() <= 0:
                    raise RuntimeError('Search returned an empty training policy')
                samples.append((env.observation().astype(np.float16), indices,
                                probabilities, team))
            # More training diversity than the referee's short opening warmup.
            temperature = 1.0 if env.action <= 8 else (0.5 if env.action <= 24 else 0.0)
            action = mcts.select_action_index(root, temperature, rng=rng)
        if action is None:
            raise RuntimeError('No legal action in a nonterminal game')
        env.step(action)
    return game_id, [(o, idx, p, 1.0 if t == env.winner else -1.0)
                     for o, idx, p, t in samples]


def pool(args, initializer, initargs):
    # Spawn avoids forking a process after PyTorch has started CPU threads.
    return ProcessPoolExecutor(max_workers=args.workers,
                               mp_context=multiprocessing.get_context('spawn'),
                               initializer=initializer, initargs=initargs)


def collect_batch(args, start, count, current_path, rivals):
    started = time.monotonic()
    records = []
    jobs = [(i, args.seed + i * 1009, args.simulations)
            for i in range(start, start + count)]
    with pool(args, worker_init, (current_path, rivals)) as executor:
        for done, (game_id, samples) in enumerate(executor.map(training_game, jobs), 1):
            records.extend((game_id, *sample) for sample in samples)
            if done % max(5, count // 10) == 0 or done == count:
                elapsed = time.monotonic() - started
                rate = done / max(elapsed, 0.001)
                print(f'Games {start+done}/{args.games} | batch {done}/{count} | '
                      f'{rate:.2f} games/s | batch ETA {(count-done)/rate:.0f}s', flush=True)
    return records


def save_replay(path, records, scene):
    n = len(records)
    width = max(len(row[2]) for row in records)
    observations = np.empty((n, 16, 8, 8), np.float16)
    indices = np.zeros((n, width), np.int64)
    probabilities = np.zeros((n, width), np.float32)
    valid = np.zeros((n, width), bool)
    games = np.empty(n, np.int64)
    values = np.empty(n, np.float32)
    for i, (game, obs, idx, prob, value) in enumerate(records):
        games[i], observations[i], values[i] = game, obs, value
        indices[i, :len(idx)] = idx
        probabilities[i, :len(idx)] = prob
        valid[i, :len(idx)] = True
    np.savez_compressed(path, observations=observations, indices=indices,
                        probabilities=probabilities, valid=valid, games=games,
                        values=values, scene=scene)


def load_replay(path, scene):
    with np.load(path) as data:
        if not np.array_equal(data['scene'], scene):
            raise RuntimeError('Replay belongs to a different scene')
        return [(int(g), o.copy(), idx[mask].astype(np.int16), prob[mask].copy(), float(v))
                for g, o, idx, mask, prob, v in zip(
                    data['games'], data['observations'], data['indices'],
                    data['valid'], data['probabilities'], data['values'])]


def recover_records(replay_path, scene, completed, count):
    records = load_replay(replay_path, scene)
    pending_games = {row[0] for row in records if row[0] >= completed}
    expected_games = set(range(completed, completed + count))
    if pending_games != expected_games:
        raise RuntimeError('Pending replay does not match this checkpoint/batch size. '
                           'Resume without --recover-pending to regenerate the batch.')
    return records


def fit(args, replay_path, weights_in, weights_out, iteration):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed + iteration)
    model = make_model()
    if weights_in:
        model.load_state_dict(torch.load(weights_in, weights_only=True, map_location='cpu'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    with np.load(replay_path) as data:
        x = torch.from_numpy(data['observations'].astype(np.float32))
        idx = torch.from_numpy(data['indices'])
        probability = torch.from_numpy(data['probabilities'])
        valid = torch.from_numpy(data['valid'])
        value = torch.from_numpy(data['values'])
        # Fixed game-level membership across iterations; no sample leakage.
        validation = data['games'] % 10 == 0
    if not validation.any() or validation.all():
        raise RuntimeError('Replay needs both training and validation games')
    train_ids = torch.from_numpy(np.flatnonzero(~validation))
    val_ids = torch.from_numpy(np.flatnonzero(validation))
    best = float('inf')

    def loss(batch):
        policy, prediction = model(x[batch])
        legal = policy.gather(1, idx[batch]).masked_fill(~valid[batch], -1e9)
        policy_loss = -(probability[batch] * F.log_softmax(legal, dim=1)).sum(1).mean()
        return policy_loss + F.mse_loss(prediction, value[batch])

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        order = train_ids[torch.randperm(len(train_ids))]
        for batch in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(batch_loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            score = sum(float(loss(b)) * len(b) for b in val_ids.split(args.batch_size)) / len(val_ids)
        if score < best:
            best = score
            torch.save(model.state_dict(), weights_out)
        print(f'Iteration {iteration}, epoch {epoch}/{args.epochs} | '
              f'train={total/len(train_ids):.4f} validation={score:.4f}', flush=True)
    model.load_state_dict(torch.load(weights_out, weights_only=True, map_location='cpu'))
    return model.eval()


def arena_init(candidate, champion):
    global _ARENA_CANDIDATE, _ARENA_CHAMPION
    from colony_one.onnx_runner import OnnxEvaluator
    _ARENA_CANDIDATE = OnnxEvaluator(candidate) if candidate else teacher
    _ARENA_CHAMPION = OnnxEvaluator(champion) if champion else None


def arena_game(job):
    game_id, seed = job
    env = ColonyOneEnv()
    config = mcts.MCTSConfig(simulations=100)
    team = constants.CYAN if game_id % 2 == 0 else constants.MAGENTA
    # Each adjacent pair shares a seed and swaps colours.
    candidate_rng = np.random.default_rng(seed)
    champion_rng = np.random.default_rng(seed + 1)
    greedy = GreedyAgent(seed=seed + 2)
    while not env.terminated:
        ours = env.current_team == team
        evaluator = _ARENA_CANDIDATE if ours else _ARENA_CHAMPION
        if evaluator is None:
            action = greedy.select(env.board, env.current_team, env.actions_left, env.round_number)
        else:
            action = mcts.search_action(
                env.board, env.current_team, env.actions_left, env.round_number,
                evaluator, config, rng=candidate_rng if ours else champion_rng,
            )
        if action is None:
            raise RuntimeError('No legal action in arena')
        env.step(action)
    return team, env.winner == team


def arena(args, candidate, champion, iteration):
    wins = 0
    colours = {constants.CYAN: [0, 0], constants.MAGENTA: [0, 0]}
    jobs = [(i, args.seed + 10000000 + iteration * 10000 + (i // 2) * 11)
            for i in range(args.arena_games)]
    with pool(args, arena_init, (candidate, champion)) as executor:
        for done, (team, won) in enumerate(executor.map(arena_game, jobs), 1):
            wins += won
            colours[team][0] += int(won)
            colours[team][1] += 1
            if done % 10 == 0 or done == args.arena_games:
                print(f'Arena {done}/{args.arena_games}: candidate {wins} wins', flush=True)
    print(f'Arena result {wins}/{args.arena_games}; colours={colours}', flush=True)
    return wins


def validate_export(path, directory):
    import colony_one
    validator = Path(colony_one.__file__).resolve().parent.parent / 'tools/validate_submission.py'
    if not validator.is_file():
        raise RuntimeError(f'Official validator not found: {validator}')
    result = subprocess.run([sys.executable, str(validator), str(path)],
                            capture_output=True, text=True, timeout=60)
    (directory / 'validation.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Official validation rejected the candidate. See '
                           f'{directory / "validation.log"}\n{result.stdout[-1800:]}')
    print('Official validator: ACCEPTED', flush=True)


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=100000, help='Total training-game budget including bootstrap')
    parser.add_argument('--batch-games', type=int, default=500)
    parser.add_argument('--bootstrap-games', type=int, default=500)
    parser.add_argument('--simulations', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--replay-positions', type=int, default=50000)
    parser.add_argument('--arena-games', type=int, default=40)
    parser.add_argument('--promote-rate', type=float, default=0.55)
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--init', type=Path, help='Optional weights.pt from the previous trainer')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recover-pending', action='store_true',
                        help='With --resume, re-export the saved pending candidate without regenerating games or retraining')
    parser.add_argument('--data-only', action='store_true', help='Generate one heuristic batch without torch/ONNX')
    args = parser.parse_args()
    if args.recover_pending and (not args.resume or args.data_only):
        parser.error('--recover-pending requires --resume and cannot use --data-only')
    if min(args.games, args.batch_games, args.bootstrap_games) < 20:
        parser.error('Use at least 20 games per budget/batch/bootstrap')
    if min(args.epochs, args.workers, args.threads, args.batch_size) < 1 or args.simulations < 8:
        parser.error('Counts must be positive; simulations >= 8')
    if args.arena_games < 20 or args.arena_games % 2 or not 0.5 < args.promote_rate <= 1:
        parser.error('Arena games must be even and >=20; promote rate must exceed 0.5')
    if args.replay_positions < 1000 or args.learning_rate <= 0:
        parser.error('Use replay positions >=1000 and positive learning rate')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    scene = ColonyOneEnv().observation()
    if not np.array_equal(scene[:12].sum(axis=(1, 2)), np.ones(12)):
        parser.error('Wrong scene: expected one of all six piece types for each team')
    scene_hash = hashlib.sha256(scene.tobytes()).hexdigest()
    state_path = args.out / 'state.json'
    state = None
    if args.resume:
        if not state_path.is_file():
            parser.error('No saved state to resume')
        state = json.loads(state_path.read_text())
        if state['scene_hash'] != scene_hash:
            parser.error('Scene changed since checkpoint; use a separate output directory')
        if state['seed'] != args.seed:
            parser.error('Keep --seed unchanged when resuming')
    elif state_path.exists():
        parser.error('Run already exists; use --resume or another --out directory')
    records = deque(maxlen=args.replay_positions)
    if state and state['replay']:
        records.extend(load_replay(state['replay'], scene))
    completed = state['completed_games'] if state else 0
    iteration = state['iteration'] if state else 0
    champion_pt = state['champion_pt'] if state else None
    champion_onnx = state['champion_onnx'] if state else None
    rivals = state.get('rivals', []) if state else []
    if not state and args.init:
        import torch
        torch.set_num_threads(args.threads)
        initial = make_model()
        initial.load_state_dict(torch.load(args.init, weights_only=True, map_location='cpu'))
        initial.eval()
        champion_pt = str(args.out / 'initial.pt')
        champion_onnx = str(args.out / 'initial.onnx')
        torch.save(initial.state_dict(), champion_pt)
        export(initial, Path(champion_onnx))
        validate_export(Path(champion_onnx), args.out)
    print(f'Level 4 | resume at {completed}/{args.games} games | '
          f'replay cap {args.replay_positions} positions', flush=True)
    recover_pending = args.recover_pending
    while completed < args.games:
        iteration += 1
        directory = args.out / f'iteration-{iteration:05d}'
        directory.mkdir(exist_ok=True)
        bootstrap = champion_pt is None
        count = min(args.bootstrap_games if bootstrap else args.batch_games, args.games - completed)
        started = time.monotonic()
        replay_path = directory / 'replay.npz'
        candidate_pt = directory / 'candidate.pt'
        candidate_onnx = directory / 'candidate.onnx'
        if recover_pending:
            import torch
            torch.set_num_threads(args.threads)
            if not candidate_pt.is_file() or not replay_path.is_file():
                raise RuntimeError('Pending weights/replay missing; resume without --recover-pending.')
            recovered = recover_records(replay_path, scene, completed, count)
            records.clear()
            records.extend(recovered)
            model = make_model()
            model.load_state_dict(torch.load(candidate_pt, weights_only=True, map_location='cpu'))
            model.eval()
            recover_pending = False
            print(f'Recovering iteration {iteration}: saved candidate and {count} generated games; '
                  'skipping data generation and training.', flush=True)
        else:
            batch = collect_batch(args, completed, count, champion_onnx, rivals)
            records.extend(batch)
            save_replay(replay_path, records, scene)
            if args.data_only:
                print(f'Data-only check complete: {len(records)} positions; {replay_path}', flush=True)
                return
            model = fit(args, replay_path, champion_pt, candidate_pt, iteration)
        export(model, candidate_onnx)
        del model
        validate_export(candidate_onnx, directory)
        wins = arena(args, str(candidate_onnx), champion_onnx, iteration)
        # Bootstrap is the first usable network. Subsequent promotions require
        # beating the current champion on a balanced, fresh-seeded arena.
        promote = bootstrap or wins >= math.ceil(args.promote_rate * args.arena_games)
        if promote:
            if champion_onnx:
                rivals = (rivals + [champion_onnx])[-4:]
            champion_pt, champion_onnx = str(candidate_pt), str(candidate_onnx)
            shutil.copyfile(champion_pt, args.out / 'best.pt')
            shutil.copyfile(champion_onnx, args.out / 'best.onnx')
            print('PROMOTED: best.onnx updated locally.', flush=True)
        else:
            print('NOT PROMOTED: keeping the previous best.onnx.', flush=True)
        completed += count
        row = {'iteration': iteration, 'completed_games': completed, 'wins': wins,
               'arena_games': args.arena_games, 'promoted': promote,
               'seconds': round(time.monotonic() - started, 2)}
        new_state = {**row, 'scene_hash': scene_hash, 'seed': args.seed,
                     'champion_pt': champion_pt, 'champion_onnx': champion_onnx,
                     'rivals': rivals, 'replay': str(replay_path),
                     'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        previous_replay = Path(state['replay']) if state and state['replay'] else None
        atomic_json(state_path, new_state)
        with (args.out / 'history.jsonl').open('a') as history:
            history.write(json.dumps(row) + '\n')
        # Keep the latest replay only; old models remain available as rivals.
        if previous_replay and previous_replay != replay_path:
            previous_replay.unlink(missing_ok=True)
        state = new_state
        remaining_cycles = math.ceil((args.games - completed) / args.batch_games)
        print(f'Checkpoint saved. Completed {completed}/{args.games}; '
              f'rough remaining time {remaining_cycles * row["seconds"] / 3600:.2f} hours '
              '(based on this cycle; excludes interruptions).', flush=True)
    print(f'DONE. Best locally accepted model: {args.out / "best.onnx"}', flush=True)


if __name__ == '__main__':
    main()
