import torch
import tempfile
import onnx
import onnxruntime as ort
import numpy as np
import time
from tqdm import tqdm


class DummyModel(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(DummyModel, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, h_dim, bias=False)
        self.fc2 = torch.nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.relu(x)
        return x


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
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

        # onnx.save(onnx_model, f"model.onnx")

        opset_import = onnx_model.opset_import
        has_custom_domain = False
        for opset in opset_import:
            if opset.domain == "olive.triton_fusion":
                has_custom_domain = True
        if not has_custom_domain:
            opset_import.extend([onnx.helper.make_opsetid("olive.triton_fusion", 1)])

        # change the type of the op to TritonMatMul
        changed = 0
        for node in onnx_model.graph.node:
            if node.op_type == "MatMul":
                node.domain = "olive.triton_fusion"
                node.op_type = "TritonMatMul"
                changed += 1
        print(f"Changed {changed} nodes")
        onnx.save(onnx_model, f"{tmpdir}/model_custom.onnx")

        # run the model with onnxruntime
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library("output/lib/libcustom_op.so")
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

        print("all close passed")

    print("Done")


if __name__ == "__main__":
    main()
