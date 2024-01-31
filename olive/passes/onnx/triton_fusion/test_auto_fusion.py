# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tqdm import tqdm

from olive.passes.onnx.triton_fusion.builder import Builder
from olive.passes.onnx.triton_fusion.fuser import Fusion
from olive.passes.onnx.triton_fusion.utils import DOMAIN


class DummyModel(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, h_dim, bias=False)
        self.fc2 = torch.nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return torch.sigmoid(x)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        # custom op dir
        print("Building custom op...")
        custom_op_dir = Path(tmpdir) / "custom_op"
        fusion = Fusion("fp32", "MatMul")
        builder = Builder([fusion], custom_op_dir)
        lib_path = builder.build()

        print("Exporting model...")
        # in_dim, h_dim, out_dim = 4096, 8192, 4096
        in_dim, h_dim, out_dim = 100, 200, 150
        batch_size = 16
        # seq_len = 1024
        seq_len = 12
        model = DummyModel(in_dim, h_dim, out_dim)
        dummy_input = torch.randn(batch_size, seq_len, in_dim)
        ort_inputs = {"x": dummy_input.numpy()}
        torch.onnx.export(model, dummy_input, f"{tmpdir}/model.onnx", opset_version=14, input_names=["x"])

        onnx_model = onnx.load(f"{tmpdir}/model.onnx")

        print("Modifying model...")
        opset_import = onnx_model.opset_import
        has_custom_domain = False
        for opset in opset_import:
            if opset.domain == DOMAIN:
                has_custom_domain = True
        if not has_custom_domain:
            opset_import.extend([onnx.helper.make_opsetid(DOMAIN, 1)])

        # change the type of the op to TritonMatMul
        changed = 0
        for node in onnx_model.graph.node:
            if node.op_type == "MatMul":
                node.domain = DOMAIN
                node.op_type = fusion.get_custom_op_name()
                changed += 1
        onnx.save(onnx_model, f"{tmpdir}/model_custom.onnx")
        print(f"Changed {changed} nodes")

        # run the model with onnxruntime
        print("Running modified model...")
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(str(lib_path))
        custom_session = ort.InferenceSession(
            f"{tmpdir}/model_custom.onnx", sess_options=sess_options, providers=["CUDAExecutionProvider"]
        )
        custom_outputs = custom_session.run(None, ort_inputs)

        num_iters = 10

        latencies = []
        for _ in tqdm(range(num_iters)):
            start = time.time()
            custom_outputs = custom_session.run(None, ort_inputs)
            end = time.time()
            latencies.append(end - start)

        print(f"Average latency: {np.mean(latencies)}")

        print("Running original model...")
        original_session = ort.InferenceSession(f"{tmpdir}/model.onnx", providers=["CUDAExecutionProvider"])

        original_latencies = []
        for _ in tqdm(range(num_iters)):
            start = time.time()
            original_outputs = original_session.run(None, ort_inputs)
            end = time.time()
            original_latencies.append(end - start)

        print(f"Average latency: {np.mean(original_latencies)}")

        # compare the outputs
        np.testing.assert_allclose(custom_outputs[0], original_outputs[0], atol=1e-2, rtol=0.0)

        print("All close test passed")

    print("Done")


if __name__ == "__main__":
    main()
