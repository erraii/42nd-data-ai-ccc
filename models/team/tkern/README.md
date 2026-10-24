# Tobias Kern — training path

Part of team **42Heilbronn_GTA6** (Eray Çakıray and Tobias Kern).
This is a second training path, built on Tobias's own machine with Claude Code
on top of the provided game engine. It used the team's self-play models as
opponents and ensemble members, so the two paths fed into each other.

## Training path

- **Tactical oracle teacher:** rebuilds the board from the 16 input planes and
  searches the side to move's remaining actions this turn for a Kernel capture,
  firewall or endgame win, or the best material gain. A null-move check estimates
  what the opponent could take next turn if we passed; the rest of the score is a
  hand-written heuristic (piece values taken from the supplied GreedyAgent).
  Locally it won 95% against the Greedy bot on Level 3 (40 games, Kernel-tactics
  version) and 96% on Level 4 (24 games).
- **Distillation (v1):** about 1.7M oracle-labelled positions (value targets) plus
  search visit counts (policy targets), trained into a 6-block × 64-channel
  residual network. Locally the oracle won 17% of 24 games against v1.
- **Self-play (v2):** v1, searching at 200 simulations, played itself and the
  team's self-play models (at 100). v2 was fine-tuned from v1 on that data mixed
  50/50 with oracle data, and beat v1 in 74% of 100 local games.
- **Ensemble:** a 50/50 average of v2 and `best_6` in a single ONNX file,
  evaluated as a submission candidate. Models were uploaded and played live
  against other teams' evolving models; the final upload was `best_8` from the
  main self-play run.

All percentages are local head-to-head results on small samples, not tournament
performance.

## Files

| File | Contents |
|---|---|
| `train/heuristic.py` | Static evaluation: material, Kernel ring position, adjacent guards |
| `train/oracle.py` | Tactical search, null-move threat check, value combination |
| `train/gen_data.py` | Oracle game generation for v1 training data |
| `train/train_net.py` | Residual network training and ONNX export |
| `train/gen_selfplay.py` | Self-play data generation for v2 |
| `train/ensemble.py` | 50/50 ensemble export into a single ONNX file |
| `train/arena.py` | Local head-to-head matches under the referee MCTS (source of the percentages above) |
| `model_tobias.onnx` | v2 -> v1 fine-tuned on self-play and oracle data (6 × 64 residual network) |
| `model_synthesis.onnx` | Ensemble of v2 and `best_6` |

Training data (`*.npz`) and PyTorch weights (`*.pt`) are not included because of
their size.

## Running the scripts

The scripts expect the layout of the provided `colony-one-env` package: `train/`
sits next to `colony_one/` and imports it from the parent directory. Install the
environment as described in its README, plus PyTorch for training
(`pip install torch onnxscript`). Some default paths, such as the opponent models
in `gen_data.py` and the examples in `gen_selfplay.py`, point to files outside
this folder; pass your own paths via the command-line options.

```bash
python train/gen_data.py --games 1500 --out train/data4            # oracle data
python train/train_net.py --data train/data4_*.npz --out models/v1.onnx --epochs 3
python train/gen_selfplay.py --model models/v1.onnx --games 1000 --out train/sp1
python train/train_net.py --init models/v1.pt --data train/data4_*.npz \
    --selfplay train/sp1_*.npz --epochs 2 --steps-per-epoch 800 --lr 3e-4 \
    --policy-weight 0.5 --out models/v2.onnx
python train/ensemble.py models/v2.onnx best_6.onnx --out model_synthesis.onnx
python train/arena.py models/v2.onnx models/v1.onnx --games 100
```
