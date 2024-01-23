from pathlib import Path
from olive.common.utils import run_subprocess

import triton


def compile_kernel(dir, signature, kernel_name, out_name, out_path, num_stages, num_warps, grid, kernel_path):
    compiler_path = Path(triton.tools.__path__[0]) / "compile.py"

    run_subprocess(
        f"python {compiler_path} -n {kernel_name} --signature {signature} --out-name {out_name} -o {out_path} -ns"
        f" {num_stages} -w {num_warps} -g {grid} {kernel_path}",
        check=True,
        cwd=dir,
    )


def compile_matmul_kernel(dir, dtype, BM, BN, BK, GM):
    sig = f"'*{dtype}, *{dtype}, *{dtype}, i32, i32, i32, {BM}, {BN}, {BK}, {GM}'"
    name = f"matmul_{dtype}"
    grid = f"'((M + {BM -1})/{BM}) * ((N + {BN - 1})/{BN}), 1, 1'"
    compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name="matmul_kernel",
        out_name=name,
        out_path=name,
        num_stages=5,
        num_warps=2,
        grid=grid,
        kernel_path=Path(__file__).parent / "kernels.py",
    )


def link_matmul_kernels(dir, dtype):
    linker_path = Path(triton.tools.__path__[0]) / "link.py"

    # link all desired configs
    h_files = [str(file) for file in Path(dir).glob(f"matmul_{dtype}*.h")]
    run_subprocess(f"python {linker_path} {' '.join(h_files)} -o matmul_{dtype} -o matmul_{dtype}", check=True, cwd=dir)

    # need to add extern C to the header file to avoid name mangling
    # header is used in c++ code also
    extern_c_start = ["#ifdef __cplusplus\n", 'extern "C" {\n', "#endif\n", "\n"]
    extern_c_end = ["\n", "#ifdef __cplusplus\n", "}\n", "#endif\n"]

    lines = []
    with open(dir / f"matmul_{dtype}.h", "r") as f:
        lines = f.readlines()
    lines = lines[:2] + extern_c_start + lines[2:] + extern_c_end
    with open(dir / f"matmul_{dtype}.h", "w") as f:
        f.writelines(lines)


def create_triton_kernels(dir, dtype):
    dir = Path(dir).resolve()
    dir.mkdir(parents=True, exist_ok=True)

    BM, BN, BK, GM = 32, 64, 32, 8
    compile_matmul_kernel(dir, dtype, BM, BN, BK, GM)
    link_matmul_kernels(dir, dtype)
