import torch
import tempfile
import onnx
import onnxruntime as ort


class DummyModel(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(DummyModel, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel(10, 20, 30)
        torch.onnx.export(model, torch.randn(12, 10, 10), f"{tmpdir}/model.onnx", opset_version=14)

        onnx_model = onnx.load(f"{tmpdir}/model.onnx")

        # change the type of the op to TritonMatMul
        changed = 0
        for node in onnx_model.graph.node:
            if node.op_type == "MatMul":
                node.op_type = "TritonMatMul"
                changed += 1
        print(f"Changed {changed} nodes")
        onnx.save(onnx_model, f"{tmpdir}/model_custom.onnx")

        # run the model with onnxruntime
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library("compiled/libcustom_op.so")
        ort_session = ort.InferenceSession(
            f"{tmpdir}/model_custom.onnx", sess_options=sess_options, providers=["CUDAExecutionProvider"]
        )
        ort_inputs = {ort_session.get_inputs()[0].name: torch.randn(1, 10).cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        print(ort_outputs)


if __name__ == "__main__":
    main()
