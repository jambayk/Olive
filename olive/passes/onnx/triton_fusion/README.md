# Triton Fused Kernels as ORT Custom Ops

## Install torch and triton
This requires latest nightly build of triton.
```bash
pip install torch
pip uninstall -y triton triton-nightly
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

## Install onnxruntime
Requires latest nightly build of ort-nightly-gpu
```bash
pip uninstall -y onnxruntime-gpu onnxruntime ort-nightly-gpu ort-nightly
pip install ort-nightly-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

## Clone ONNX Runtime repository
```bash
git clone https://github.com/microsoft/onnxruntime.git
# set ONNXRUNTIME_DIR to the path of the cloned repository
export ONNXRUNTIME_DIR=$PWD/onnxruntime
```

Also requires the path to cuda to be set using `CUDA_HOME` environment variable. Only tested with CUDA 12.2

## Run the example
```bash
python test_triton_fusion.py
```
