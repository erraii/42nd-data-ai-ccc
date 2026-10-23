#!/usr/bin/env python3
"""CPU starter for the official Colony: One Level 3 environment.

Run from the folder containing colony_one/ and tools/.
Data: MCTS teacher vs GreedyAgent (2/3) and teacher self-play (1/3).
Learning: masked visit-count imitation + final game-result value regression.
The teacher is a hand-written heuristic, not an optimal solver.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

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


def generate_game(job):
    game_id, seed, simulations = job
    rng = np.random.default_rng(seed)
    env = ColonyOneEnv(seed=seed)
    greedy = GreedyAgent(seed=seed + 1)
    # Alternate colours in teacher-vs-greedy games; every third is self-play.
    teacher_team = constants.CYAN if game_id % 2 == 0 else constants.MAGENTA
    config = mcts.MCTSConfig(simulations=simulations)
    records = []
    while not env.terminated:
        if game_id % 3 == 2 or env.current_team == teacher_team:
            root = mcts.run(
                env.board, env.current_team, env.actions_left,
                env.round_number, teacher, config, rng=rng,
            )
            legal = env.legal_action_indices().astype(np.int16)
            target = mcts.policy_target(root)[legal]
            if target.sum() <= 0:
                raise RuntimeError("MCTS produced no visits")
            records.append((env.observation().astype(np.float16), legal, target, env.current_team))
            temperature = mcts.warmup_temperature(env.action)
            action = mcts.select_action_index(root, temperature, rng=rng)
        else:
            action = greedy.select(env.board, env.current_team, env.actions_left, env.round_number)
        if action is None:
            raise RuntimeError("No legal action in a nonterminal game")
        env.step(action)
    return game_id, records, env.winner, teacher_team


def collect(args, destination):
    jobs = [(i, args.seed + i * 1009, args.simulations) for i in range(args.games)]
    observations, legal_rows, targets, values, game_ids = [], [], [], [], []
    wins = matches = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for done, (game_id, records, winner, team) in enumerate(executor.map(generate_game, jobs), 1):
            if game_id % 3 != 2:
                matches += 1
                wins += winner == team
            for obs, legal, target, acting_team in records:
                observations.append(obs)
                legal_rows.append(legal)
                targets.append(target)
                values.append(1.0 if acting_team == winner else -1.0)
                game_ids.append(game_id)
            if done % max(5, args.games // 20) == 0 or done == args.games:
                print(f"Data: {done}/{args.games} games | {len(values)} positions | "
                      f"teacher vs greedy: {wins}/{matches} | {time.monotonic()-started:.1f}s", flush=True)
    width = max(map(len, legal_rows))
    indices = np.zeros((len(values), width), np.int64)
    probabilities = np.zeros((len(values), width), np.float32)
    valid = np.zeros((len(values), width), bool)
    for i, (legal, target) in enumerate(zip(legal_rows, targets)):
        indices[i, :len(legal)] = legal
        probabilities[i, :len(legal)] = target
        valid[i, :len(legal)] = True
    np.savez_compressed(
        destination, observations=np.stack(observations), indices=indices,
        probabilities=probabilities, valid=valid,
        values=np.array(values, np.float32), games=np.array(game_ids),
        scene=ColonyOneEnv().observation(),
    )
    print(f"Saved dataset: {destination}", flush=True)


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
            value = torch.clamp(self.value(hidden).squeeze(1), -0.999999, 0.999999)
            return self.policy(hidden), value

    return PolicyValue()


def train(args, dataset, checkpoint):
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data = np.load(dataset)
    x = torch.from_numpy(data['observations'].astype(np.float32))
    indices = torch.from_numpy(data['indices'])
    targets = torch.from_numpy(data['probabilities'])
    valid = torch.from_numpy(data['valid'])
    values = torch.from_numpy(data['values'])
    # Split entire games, so positions from a validation game never train.
    rng = np.random.default_rng(args.seed)
    games = rng.permutation(np.unique(data['games']))
    validation_games = games[:max(1, len(games) // 5)]
    is_validation = np.isin(data['games'], validation_games)
    train_ids = torch.from_numpy(np.flatnonzero(~is_validation))
    validation_ids = torch.from_numpy(np.flatnonzero(is_validation))
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = float('inf')

    def losses(batch):
        logits, predicted_value = model(x[batch])
        legal_logits = logits.gather(1, indices[batch]).masked_fill(~valid[batch], -1e9)
        policy_loss = -(targets[batch] * F.log_softmax(legal_logits, dim=1)).sum(1).mean()
        value_loss = F.mse_loss(predicted_value, values[batch])
        return policy_loss + value_loss

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_ids[torch.randperm(len(train_ids))]
        total = 0.0
        for batch in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = losses(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            validation_loss = sum(float(losses(b)) * len(b) for b in validation_ids.split(args.batch_size)) / len(validation_ids)
        if validation_loss < best:
            best = validation_loss
            torch.save(model.state_dict(), checkpoint)
        print(f"Epoch {epoch}/{args.epochs} | train {total/len(train_ids):.4f} | "
              f"validation {validation_loss:.4f} | best {best:.4f}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return model.eval()


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


def evaluate(path, games, seed, simulations=60, both_colours=False, opponent_path=None):
    from colony_one.onnx_runner import OnnxEvaluator

    evaluator = OnnxEvaluator(path)
    other_evaluator = OnnxEvaluator(opponent_path) if opponent_path else None
    opponent_name = str(opponent_path) if opponent_path else 'GreedyAgent'
    config = mcts.MCTSConfig(simulations=simulations)
    wins = 0
    by_colour = {constants.CYAN: [0, 0], constants.MAGENTA: [0, 0]}
    for i in range(games):
        env = ColonyOneEnv()
        opponent = GreedyAgent(seed=seed + 2 * i)
        rng = np.random.default_rng(seed + 2 * i + 1)
        opponent_rng = np.random.default_rng(seed + 2 * i + 500000)
        model_team = constants.MAGENTA if both_colours and i % 2 else constants.CYAN
        while not env.terminated:
            if env.current_team == model_team:
                action = mcts.search_action(
                    env.board, env.current_team, env.actions_left,
                    env.round_number, evaluator, config, rng=rng,
                )
            elif other_evaluator is not None:
                action = mcts.search_action(
                    env.board, env.current_team, env.actions_left,
                    env.round_number, other_evaluator, config, rng=opponent_rng,
                )
            else:
                action = opponent.select(env.board, env.current_team, env.actions_left, env.round_number)
            if action is None:
                raise RuntimeError('No action during evaluation')
            env.step(action)
        won = env.winner == model_team
        wins += won
        by_colour[model_team][0] += int(won)
        by_colour[model_team][1] += 1
        print(f"Evaluation {i+1}/{games}: model={model_team}, winner={env.winner} | "
              f"model wins {wins}/{i+1}", flush=True)
    print(f"FINAL: {wins}/{games} wins against {opponent_name} (MCTS {simulations}).", flush=True)
    print(f"By colour: {by_colour} (wins, games)", flush=True)
    return wins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--simulations', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-games', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=Path('runs/colony'))
    parser.add_argument('--reuse-data', action='store_true')
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--export-only', action='store_true',
                        help='Re-export weights.pt from --out, without retraining')
    parser.add_argument('--evaluate', type=Path, help='Evaluate an existing ONNX file, without training')
    parser.add_argument('--level4', action='store_true',
                        help='Require the six-piece scene; evaluate both colours at MCTS 100')
    parser.add_argument('--eval-opponent', type=Path,
                        help='Also compare against this older ONNX model, using the same search budget')
    args = parser.parse_args()
    if min(args.games, args.epochs, args.workers, args.threads, args.batch_size, args.eval_games) < 1 or args.games < 5 or args.simulations < 8:
        parser.error('Use games >= 5, simulations >= 8 and positive counts.')
    initial = ColonyOneEnv().observation()
    if args.level4:
        counts = initial[:12].sum(axis=(1, 2))
        if not np.array_equal(counts, np.ones(12)):
            parser.error('Level 4 needs one of all six pieces per team. Check colony_one/scene.json.')
        if args.eval_games % 2:
            parser.error('Use an even --eval-games count to test both colours equally.')
        print('Level 4 scene verified: 6 pieces per team; evaluation MCTS=100, both colours.', flush=True)
    if args.eval_opponent and not args.eval_opponent.is_file():
        parser.error(f'Missing opponent: {args.eval_opponent}')

    def run_evaluation(path):
        simulations = 100 if args.level4 else 60
        wins = evaluate(path, args.eval_games, args.seed + 100000,
                        simulations, args.level4)
        if args.eval_opponent:
            evaluate(path, args.eval_games, args.seed + 200000,
                     simulations, args.level4, args.eval_opponent)
        return wins

    if args.evaluate:
        run_evaluation(args.evaluate)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        import torch
        torch.set_num_threads(args.threads)
        checkpoint = args.out / 'weights.pt'
        if not checkpoint.is_file():
            parser.error(f'Missing checkpoint: {checkpoint}')
        model = make_model()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        path = args.out / 'model.onnx'
        export(model, path)
        run_evaluation(path)
        return
    dataset = args.out / 'dataset.npz'
    if args.reuse_data:
        if not dataset.is_file():
            parser.error(f'Missing dataset: {dataset}')
        with np.load(dataset) as old_data:
            if 'scene' not in old_data or not np.array_equal(old_data['scene'], initial):
                parser.error('Dataset scene differs or is unknown. Generate fresh data without --reuse-data.')
    else:
        collect(args, dataset)
    if args.data_only:
        return
    model = train(args, dataset, args.out / 'weights.pt')
    path = args.out / 'model.onnx'
    export(model, path)
    wins = run_evaluation(path)
    (args.out / 'results.json').write_text(json.dumps({
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'wins': wins, 'evaluation_games': args.eval_games,
    }, indent=2) + '\n')
    print(f'Next: python tools/validate_submission.py {path}', flush=True)


if __name__ == '__main__':
    main()
