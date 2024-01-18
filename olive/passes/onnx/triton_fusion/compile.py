from pathlib import Path
from olive.common.utils import run_subprocess

import triton


def compile_kernel(dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path):
    compiler_path = Path(triton.tools.__path__[0]) / "compile.py"

    run_subprocess(
        f"python {compiler_path} -n {kernel_name} --signature {signature} --out-name {out_name} -o {out_path} -w"
        f" {num_warps} -g {grid} {kernel_path}",
        check=True,
        cwd=dir,
    )


def compile_matmul_kernel(dir, dtype, BM, BN, BK, GM):
    sig = f"'*{dtype}, *{dtype}, *{dtype}, i32, i32, i32, i32, i32, i32, i32, i32, i32, {BM}, {BN}, {BK}, {GM}'"
    name = f"matmul_{dtype}"
    grid = f"'((M + {BM -1})/{BM}) * ((N + {BM - 1})/{BN}), 1, 1'"
    compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name="matmul_kernel",
        out_name=name,
        out_path=name,
        num_warps=8,
        grid=grid,
        kernel_path=Path(__file__).parent / "kernels.py",
    )


def link_matmul_kernels(dir, dtype):
    linker_path = Path(triton.tools.__path__[0]) / "link.py"

    # link all desired configs
    h_files = [str(file) for file in Path(dir).glob(f"matmul_{dtype}*.h")]
    run_subprocess(f"python {linker_path} {' '.join(h_files)} -o matmul_{dtype} -o matmul_{dtype}", check=True, cwd=dir)


if __name__ == "__main__":
    dtype = "fp32"
    dir = Path("jk").resolve()
    dir.mkdir(exist_ok=True, parents=True)
    compile_matmul_kernel(dir, dtype, 128, 256, 64, 8)
    link_matmul_kernels(dir, dtype)
