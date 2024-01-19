from olive.common.utils import run_subprocess
from pathlib import Path
import os
import shutil


def get_cuda_include_dir():
    if not os.environ.get("CUDA_HOME"):
        raise RuntimeError("CUDA_HOME not set")

    return Path(os.environ["CUDA_HOME"]) / "include"


def get_cuda_lib_dir():
    if not os.environ.get("CUDA_HOME"):
        raise RuntimeError("CUDA_HOME not set")

    return Path(os.environ["CUDA_HOME"]) / "lib64"


def get_ort_include_dir():
    if not os.environ.get("ONNXRUNTIME_DIR"):
        raise RuntimeError("ONNXRUNTIME_DIR not set")

    return Path(os.environ["ONNXRUNTIME_DIR"]) / "include" / "onnxruntime"


def get_ort_api_include_dir():
    if not os.environ.get("ONNXRUNTIME_DIR"):
        raise RuntimeError("ONNXRUNTIME_DIR not set")

    return Path(os.environ["ONNXRUNTIME_DIR"]) / "include" / "onnxruntime" / "core" / "session"


# def gen_triton_kernel_lib(src_dir: str, out_dir: str):
#     src_files = [str(file) for file in Path(src_dir).resolve().glob("*.c")]
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     run_subprocess(f"gcc  -c -fPIC -I {get_cuda_include_dir()} {' '.join(src_files)}", check=True, cwd=out_dir)

#     obj_files = [str(file) for file in Path(out_dir).resolve().glob("*.o")]
#     run_subprocess(
#         f"gcc -shared -o libtriton_kernels.so {' '.join(obj_files)} -l cuda",
#         check=True,
#         cwd=out_dir,
#     )

#     return [str(file) for file in Path(out_dir).resolve().glob("*.so")]


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
    run_subprocess(
        f"gcc -shared -o libcustom_op.so {' '.join(triton_kernel_objs)} {' '.join(custom_op_objs)} -L"
        f" {Path(get_cuda_lib_dir()).resolve()} -l cuda -Xlinker --version-script {Path(version_script).resolve()}",
        check=True,
        cwd=out_dir,
    )


# def generate_custom_op_lib(custom_op_objs, triton_kernel_lib_dir, version_script, out_dir):
#     run_subprocess(
#         f"gcc -shared -o libcustom_op.so {' '.join(custom_op_objs)} -L {Path(triton_kernel_lib_dir).resolve()} -l"
#         " triton_kernels -l cuda -Xlinker --version-script"
#         f" {Path(version_script).resolve()}",
#         check=True,
#         cwd=out_dir,
#     )


if __name__ == "__main__":
    shutil.rmtree("compiled", ignore_errors=True)
    print("Generating triton kernel library")
    triton_kernel_objs = compile_triton_kernel("triton_kernels", "compiled/triton_kernels")

    print("Compiling custom ops")
    custom_op_objs = compile_custom_ops("custom_op", "compiled/custom_op", "triton_kernels")

    print("Generating custom op library")
    generate_custom_op_lib(triton_kernel_objs, custom_op_objs, "custom_op/custom_op_library.lds", "compiled")
