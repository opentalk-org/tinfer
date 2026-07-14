import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch

from tools.styletts2_model_scripts.model_graphs import blend_style, guided_output
from tools.styletts2_model_scripts.onnx_export import promote_weights


def test_blend_style_matches_python_batch_style_behavior() -> None:
    candidate = torch.full((2, 256), 0.25)
    reference = torch.cat((torch.full((2, 128), 1.0), torch.full((2, 128), 2.0)), dim=1)
    previous = torch.full((2, 256), 0.5)
    style, ref = blend_style(
        candidate,
        reference,
        torch.tensor([True, False]),
        previous,
        torch.tensor([True, True]),
        torch.tensor([0.6, 0.6]),
        torch.tensor([[0.25], [0.25]]),
        torch.tensor([[0.75], [0.75]]),
    )

    assert torch.allclose(ref[0], torch.full((128,), 0.85))
    assert torch.allclose(style[0], torch.full((128,), 0.8))
    assert torch.allclose(ref[1], torch.full((128,), 1.0))
    assert torch.allclose(style[1], torch.full((128,), 2.0))


def test_guidance_scale_broadcasts_per_batch_item() -> None:
    conditional = torch.ones(2, 3, 4)
    unconditional = torch.zeros(2, 3, 4)

    output = guided_output(conditional, unconditional, torch.tensor([1.0, 2.0]))

    assert torch.equal(output[0], torch.ones(3, 4))
    assert torch.equal(output[1], torch.full((3, 4), 2.0))


def test_promote_weights_turns_initializers_into_runtime_inputs() -> None:
    weight = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "weight")
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [weight],
    )
    model = helper.make_model(graph)

    weights = promote_weights(model)

    assert not model.graph.initializer
    assert [value.name for value in model.graph.input] == ["input", "weight"]
    assert np.array_equal(weights["weight"], np.array([2.0], dtype=np.float32))


def test_onnx_export_does_not_import_tensorrt() -> None:
    check = "import sys; import tools.styletts2_model_scripts.onnx_export; assert 'tensorrt' not in sys.modules"
    subprocess.run([sys.executable, "-c", check], check=True)
