"""Merge several submission ONNX models into one that averages their outputs.

    python train/ensemble.py models/v2.onnx ../latest/best_6.onnx --out models/ens.onnx
    python train/ensemble.py a.onnx b.onnx --weights 0.6 0.4 --out models/ens.onnx
    python train/ensemble.py a.onnx b.onnx --weights 0.5 0.5 --value-weights 0.7 0.3 ...

Policy logits and values are weighted averages. Averaging raw logits is the
same, after the referee's softmax over legal actions, as a weighted geometric
mean of the members' move distributions. A convex mix of values in [-1, 1]
stays in [-1, 1].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def load_member(path: Path, prefix: str) -> onnx.ModelProto:
    model = onnx.load(str(path))
    # Our own exports may still carry Constant nodes; the allow-list rejects them.
    for node in [n for n in model.graph.node if n.op_type == "Constant"]:
        model.graph.initializer.append(
            numpy_helper.from_array(numpy_helper.to_array(node.attribute[0].t),
                                    node.output[0]))
        model.graph.node.remove(node)
    return onnx.compose.add_prefix(model, prefix)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--value-weights", type=float, nargs="+",
                        help="separate weights for the value (default: --weights)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    weights = args.weights or [1.0 / len(args.models)] * len(args.models)
    if len(weights) != len(args.models):
        parser.error("one weight per model")
    weights = np.asarray(weights, dtype=np.float32) / np.sum(weights)
    value_weights = args.value_weights or weights
    if len(value_weights) != len(args.models):
        parser.error("one value weight per model")
    value_weights = np.asarray(value_weights, dtype=np.float32) / np.sum(value_weights)

    nodes, initializers = [], []
    policies, values = [], []
    for i, (path, weight, value_weight) in enumerate(
            zip(args.models, weights, value_weights, strict=True)):
        prefix = f"m{i}_"
        member = load_member(path, prefix)
        graph = member.graph
        member_input = graph.input[0].name
        # Feed every member from the shared observation input.
        nodes.append(helper.make_node("Identity", ["observation"], [member_input]))
        nodes.extend(graph.node)
        initializers.extend(graph.initializer)
        policy, value = graph.output[0].name, graph.output[1].name
        scale, value_scale = f"{prefix}weight", f"{prefix}value_weight"
        initializers.append(numpy_helper.from_array(np.array(weight, np.float32), scale))
        initializers.append(
            numpy_helper.from_array(np.array(value_weight, np.float32), value_scale))
        nodes.append(helper.make_node("Mul", [policy, scale], [f"{prefix}policy_w"]))
        nodes.append(helper.make_node("Mul", [value, value_scale], [f"{prefix}value_w"]))
        policies.append(f"{prefix}policy_w")
        values.append(f"{prefix}value_w")

    nodes.append(helper.make_node("Sum", policies, ["policy"]))
    nodes.append(helper.make_node("Sum", values, ["value_raw"]))
    # Guard against members whose value is not exactly bounded.
    nodes.append(helper.make_node("Clip", ["value_raw", "clip_lo", "clip_hi"], ["value"]))
    initializers.append(numpy_helper.from_array(np.array(-1.0, np.float32), "clip_lo"))
    initializers.append(numpy_helper.from_array(np.array(1.0, np.float32), "clip_hi"))

    graph = helper.make_graph(
        nodes, "ensemble",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
        [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB), "
          f"policy weights {np.round(weights, 3).tolist()}, "
          f"value weights {np.round(value_weights, 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
