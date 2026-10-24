"""Train the policy-value CNN on oracle data and export model.onnx.

    python train/train_net.py --data train/data4_*.npz --out model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_data import from_uint8


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class Net(nn.Module):
    def __init__(self, ch: int = 64, blocks: int = 6) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(16, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        # Policy: per from-square, 64 channels = the to-square.
        self.policy = nn.Conv2d(ch, 64, 1)
        self.v_conv = nn.Sequential(nn.Conv2d(ch, 8, 1), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(8 * 64, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.body(self.stem(x))
        p = self.policy(h)                       # (B, to=64, 8, 8)
        p = p.reshape(-1, 64, 64).transpose(1, 2)  # (B, from, to)
        policy = p.reshape(-1, 4096)
        v = self.v_conv(h).reshape(-1, 8 * 64)
        value = torch.tanh(self.v_fc(v)).reshape(-1)
        return policy, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--init", type=Path, help="state dict to start from")
    parser.add_argument("--out", type=Path, default=Path("model.onnx"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--selfplay", type=Path, nargs="*", default=[],
                        help="gen_selfplay.py shards; half of every batch comes from them")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--policy-weight", type=float, default=0.2)
    parser.add_argument("--steps-per-epoch", type=int, help="default: one pass over --data")
    args = parser.parse_args()

    torch.manual_seed(0)
    parts = [np.load(p) for p in args.data]
    # Planes stay uint8 in memory and are decoded per batch.
    planes = torch.from_numpy(np.concatenate([d["planes"] for d in parts]))
    values = torch.from_numpy(np.concatenate([d["values"] for d in parts]))
    roots = torch.from_numpy(from_uint8(np.concatenate([d["roots"] for d in parts])))
    policies = torch.from_numpy(
        np.concatenate([d["policies"] for d in parts]).astype(np.float32))
    decode = lambda batch: torch.from_numpy(from_uint8(batch.numpy()))
    print(f"{len(planes)} value positions, {len(roots)} policy positions")
    if args.selfplay:
        sp = [np.load(p) for p in args.selfplay]
        sp_planes = torch.from_numpy(np.concatenate([d["planes"] for d in sp]))
        sp_values = torch.from_numpy(np.concatenate([d["values"] for d in sp]))
        sp_policies = torch.from_numpy(
            np.concatenate([d["policies"] for d in sp]).astype(np.float32))
        print(f"{len(sp_planes)} self-play positions")

    n_val = len(planes) // 20
    perm = torch.randperm(len(planes))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net(args.channels, args.blocks)
    if args.init:
        net.load_state_dict(torch.load(args.init))
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = args.steps_per_epoch or len(tr_idx) // 512
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    for epoch in range(args.epochs):
        net.train()
        for step in range(steps_per_epoch):
            if args.selfplay:
                idx = tr_idx[torch.randint(len(tr_idx), (256,))]
                sidx = torch.randint(len(sp_planes), (256,))
                x = torch.cat([decode(planes[idx]), decode(sp_planes[sidx])])
                z = torch.cat([values[idx], sp_values[sidx]])
            else:
                idx = tr_idx[torch.randint(len(tr_idx), (512,))]
                x, z = decode(planes[idx]), values[idx]
            _, v = net(x.to(device))
            loss_v = F.mse_loss(v, z.to(device))
            pidx = torch.randint(len(roots), (128,))
            proots, target = roots[pidx], policies[pidx]
            if args.selfplay:
                sidx = torch.randint(len(sp_planes), (256,))
                proots = torch.cat([proots, decode(sp_planes[sidx])])
                target = torch.cat([target, sp_policies[sidx]])
            logits, _ = net(proots.to(device))
            target = target.to(device)
            # Only legal (visited) actions get mass; mask the rest out of the softmax.
            logits = logits.masked_fill(target <= 0, -1e4)
            loss_p = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss = loss_v + args.policy_weight * loss_p
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            preds = torch.cat([net(decode(planes[val_idx[i:i + 2048]]).to(device))[1].cpu()
                               for i in range(0, n_val, 2048)])
            z = values[val_idx]
            mse = F.mse_loss(preds, z).item()
            wins = z > 0.99
            win_acc = ((preds[wins] > 0.8).float().mean().item()
                       if wins.any() else float("nan"))
            false_win = (preds[~wins] > 0.8).float().mean().item()
        print(f"epoch {epoch + 1:2d}  val mse {mse:.4f}  win-detect {win_acc:.3f}  "
              f"false-win {false_win:.4f}  policy {loss_p.item():.3f}", flush=True)

    net = net.cpu().eval()
    torch.save(net.state_dict(), args.out.with_suffix(".pt"))
    dummy = torch.zeros(2, 16, 8, 8)
    torch.onnx.export(
        net, (dummy,), str(args.out), input_names=["observation"],
        output_names=["policy", "value"], opset_version=17, dynamo=False,
        dynamic_axes={"observation": {0: "batch"}, "policy": {0: "batch"},
                      "value": {0: "batch"}},
    )
    # The reshapes leave symbolic output dims; pin them to the named batch axis.
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(args.out))
    # The allow-list has no Constant operator: turn them into initializers.
    for node in [n for n in model.graph.node if n.op_type == "Constant"]:
        tensor = numpy_helper.from_array(
            numpy_helper.to_array(node.attribute[0].t), node.output[0])
        model.graph.initializer.append(tensor)
        model.graph.node.remove(node)
    for output, dims in zip(model.graph.output, (["batch", 4096], ["batch"]), strict=True):
        shape = output.type.tensor_type.shape
        del shape.dim[:]
        for dim in dims:
            entry = shape.dim.add()
            if isinstance(dim, str):
                entry.dim_param = dim
            else:
                entry.dim_value = dim
    onnx.checker.check_model(model)
    onnx.save(model, str(args.out))
    print(f"exported {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")

    # The exported graph must agree with the network it came from.
    import onnxruntime as ort
    sample = decode(planes[val_idx[:16]])
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    onnx_policy, onnx_value = session.run(None, {"observation": sample.numpy()})
    with torch.no_grad():
        torch_policy, torch_value = net(sample)
    print(f"onnx vs torch max diff: policy "
          f"{np.abs(onnx_policy - torch_policy.numpy()).max():.2e}, value "
          f"{np.abs(onnx_value - torch_value.numpy()).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
