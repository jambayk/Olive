# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path

from compile_triton_kernels import create_triton_kernels

from olive.common.utils import run_subprocess
from olive.passes.onnx.triton_fusion.utils import get_env_path


def get_cuda_include_dir():
    return get_env_path("CUDA_HOME") / "include"


def get_cuda_lib_dir():
    return get_env_path("CUDA_HOME") / "lib64"


def get_ort_include_dir():
    return get_env_path("ONNXRUNTIME_DIR") / "include" / "onnxruntime"


def get_ort_api_include_dir():
    return get_env_path("ONNXRUNTIME_DIR") / "include" / "onnxruntime" / "core" / "session"


def compile_triton_kernel(src_dir: str, out_dir: str):
    src_files = [str(file) for file in Path(src_dir).resolve().glob("*.c")]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_subprocess(f"gcc  -c -fPIC -I {get_cuda_include_dir()} {' '.join(src_files)}", check=True, cwd=out_dir)

    return [str(file) for file in Path(out_dir).resolve().glob("*.o")]


def compile_custom_ops(src_dir: str, out_dir: str, triton_kernels_src: str):
    src_files = [str(file) for file in Path(src_dir).resolve().glob("*.cc")]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_subprocess(
        f"gcc  -c -fPIC -I {get_cuda_include_dir()} -I {Path(triton_kernels_src).resolve()} -I"
        f" {get_ort_include_dir()} -I {get_ort_api_include_dir()} {' '.join(src_files)}",
        check=True,
        cwd=out_dir,
    )

    return [str(file) for file in Path(out_dir).resolve().glob("*.o")]


def generate_custom_op_lib(triton_kernel_objs, custom_op_objs, version_script, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    run_subprocess(
        f"gcc -shared -o libcustom_op.so {' '.join(triton_kernel_objs)} {' '.join(custom_op_objs)} -L"
        f" {Path(get_cuda_lib_dir()).resolve()} -l cuda -Xlinker --version-script {Path(version_script).resolve()}",
        check=True,
        cwd=out_dir,
    )


if __name__ == "__main__":
    output_dir = Path("output")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # create triton kernels
    create_triton_kernels("output/csrc/triton_kernels", "fp32")

    # copy over custom op src
    shutil.copytree("custom_op_src", "output/csrc/custom_op")

    # build triton kernel objects
    triton_kernel_objs = compile_triton_kernel("output/csrc/triton_kernels", "output/obj/triton_kernels")

    # build custom op objects
    custom_op_objs = compile_custom_ops("output/csrc/custom_op", "output/obj/custom_op", "output/csrc/triton_kernels")

    # generate custom op library
    generate_custom_op_lib(triton_kernel_objs, custom_op_objs, "custom_op_src/custom_op_library.lds", "output/lib")
