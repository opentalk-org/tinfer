from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).parent / "graphs"


def graph_one(
    path: Path,
    dimension: int | str,
    duration_base: int,
    output_dimension: int | str | None = None,
) -> None:
    tokens = helper.make_tensor_value_info("tokens", TensorProto.INT64, [dimension])
    declared_output = output_dimension if output_dimension is not None else dimension
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [declared_output])
    durations = helper.make_tensor_value_info("durations", TensorProto.INT64, [declared_output])
    two = helper.make_tensor("two", TensorProto.INT64, [], [2])
    base = helper.make_tensor("base", TensorProto.INT64, [], [duration_base])
    nodes = [
        helper.make_node("Cast", ["tokens"], ["features"], to=TensorProto.FLOAT),
        helper.make_node("Mod", ["tokens", "two"], ["mod"]),
        helper.make_node("Add", ["mod", "base"], ["durations"]),
    ]
    graph = helper.make_graph(nodes, "stub_graph_1_review", [tokens], [features, durations], [two, base])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


graph_one(ROOT / "graph_1_static_65.onnx", 65, 1)
graph_one(ROOT / "graph_1_excessive_duration.onnx", "tokens", 1_000)
graph_one(ROOT / "graph_1_aggregate_duration.onnx", "tokens", 100)


def graph_one_static_output(path: Path) -> None:
    tokens = helper.make_tensor_value_info("tokens", TensorProto.INT64, ["tokens"])
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [65])
    durations = helper.make_tensor_value_info("durations", TensorProto.INT64, ["tokens"])
    values = helper.make_tensor("values", TensorProto.FLOAT, [65], [1.0] * 65)
    one = helper.make_tensor("one", TensorProto.INT64, [], [1])
    nodes = [
        helper.make_node("Constant", [], ["features"], value=values),
        helper.make_node("Add", ["tokens", "one"], ["durations"]),
    ]
    graph = helper.make_graph(nodes, "static_output", [tokens], [features, durations], [one])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


graph_one_static_output(ROOT / "graph_1_static_output_65.onnx")


def graph_one_double_output(path: Path) -> None:
    tokens = helper.make_tensor_value_info("tokens", TensorProto.INT64, ["tokens"])
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["output"])
    durations = helper.make_tensor_value_info("durations", TensorProto.INT64, ["output"])
    nodes = [
        helper.make_node("Concat", ["tokens", "tokens"], ["doubled"], axis=0),
        helper.make_node("Cast", ["doubled"], ["features"], to=TensorProto.FLOAT),
        helper.make_node("Identity", ["doubled"], ["durations"]),
    ]
    graph = helper.make_graph(nodes, "double_output", [tokens], [features, durations])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


graph_one_double_output(ROOT / "graph_1_double_output.onnx")
