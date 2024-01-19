import torch
import tempfile
import onnx
import onnxruntime as ort
import numpy as np
import shutil
from pathlib import Path


class DummyModel(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(DummyModel, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, h_dim, bias=False)
        # self.fc2 = torch.nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        return x


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel(10, 20, 30)
        torch.onnx.export(model, torch.randn(12, 11, 10), f"{tmpdir}/model.onnx", opset_version=14)

        onnx_model = onnx.load(f"{tmpdir}/model.onnx")

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
        sess_options.register_custom_ops_library("compiled/libcustom_op.so")
        ort_session = ort.InferenceSession(
            f"{tmpdir}/model_custom.onnx", sess_options=sess_options, providers=["CUDAExecutionProvider"]
        )
        ort_inputs = {ort_session.get_inputs()[0].name: torch.randn(12, 11, 10).cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        original_session = ort.InferenceSession(f"{tmpdir}/model.onnx", providers=["CUDAExecutionProvider"])
        original_outputs = original_session.run(None, ort_inputs)

        outdir = "output"
        shutil.rmtree(outdir, ignore_errors=True)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{outdir}/model.pt")
        np.save(f"{outdir}/input.npy", ort_inputs[ort_session.get_inputs()[0].name])
        np.save(f"{outdir}/output.npy", ort_outputs[0])
        np.save(f"{outdir}/output_original.npy", original_outputs[0])

        print("Done")

        # compare the outputs
        np.testing.assert_allclose(ort_outputs[0], original_outputs[0], atol=1e-4, rtol=0.0)


if __name__ == "__main__":
    main()
