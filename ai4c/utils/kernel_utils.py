import os
import re
import glob
import tempfile
import subprocess

from typing import Dict, Tuple
from unittest.mock import patch


def remove_pybind_module(cuda_code: str) -> str:
    """delete 'PYBIND11_MODULE(...) {...}' block from CUDA source string."""
    pattern = re.compile(r"PYBIND11_MODULE\s*\([^)]*\)\s*\{[^{}]*\}", re.DOTALL)
    cleaned_code = re.sub(pattern, "", cuda_code)
    return cleaned_code.strip()


def extract_cuda_kernel_names(cuda_code: str):
    match = re.search(r'r?"""([\s\S]*?)"""', cuda_code)
    if not match:
        return []
    cuda_src = match.group(1)

    # remove comments
    cuda_src = re.sub(r"/\*.*?\*/", "", cuda_src, flags=re.S)
    cuda_src = re.sub(r"//.*", "", cuda_src)

    # match kernel function names
    pattern = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(", flags=re.MULTILINE)
    kernels = pattern.findall(cuda_src)

    return kernels


def extract_cuda_code(text: str):
    """Extract the last code block from the given text."""
    codeblock_seps = ["python", ""]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    codeblock_start = f"```(?:{languages_pattern})"
    pattern = re.compile(
        codeblock_start + r"\s*\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL | re.IGNORECASE
    )

    matches = list(pattern.finditer(text))
    if matches:
        last_match = matches[-1]
        code_content = last_match.group(1).rstrip()
        return code_content
    return text


def _compile_kernel(cuda_code: str) -> Tuple[bool, Dict]:
    ret = {
        "ext_filename": None,
        "ext_content": None,
        "msg": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)

        compile_log = ""
        success = True
        try:
            compile_cmd = f"python3 model_new.py"
            with patch.dict(
                os.environ,
                {
                    "TORCH_CUDA_ARCH_LIST": "8.0",
                    "TORCH_EXTENSIONS_DIR": "build",
                    "MAX_JOBS": "64",
                },
            ):
                compile_result = subprocess.run(
                    compile_cmd,
                    timeout=180,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    cwd=tmpdir,
                )
            compile_log = compile_result.stdout.decode()
            so_files = glob.glob(f"{tmpdir}/build/**/*.so")
            assert len(so_files) == 1, f"should generate 1 .so file, got {so_files}"
            with open(so_files[0], "rb") as fin:
                bin_content = fin.read()
            ret["ext_filename"] = os.path.basename(so_files[0])
            ret["ext_content"] = bin_content
            ret["msg"] = "compile success"
            success = True
        except subprocess.TimeoutExpired as e:
            success = False
            ret["msg"] = "failed: compilation timed out"
        except Exception as e:
            success = False
            ret["msg"] = f"failed: compilation error: [{e}] log: [{compile_log}]"
        return success, ret


def _compile_ext(cuda_code: str) -> Tuple[bool, Dict]:
    ret = {
        "ext_filename": None,
        "ext_content": None,
        "msg": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)

        compile_log = ""
        success = True
        try:
            compile_cmd = f"python3 model_new.py"
            with patch.dict(
                os.environ,
                {
                    "TORCH_CUDA_ARCH_LIST": "8.0",
                    "TORCH_EXTENSIONS_DIR": "build",
                    "MAX_JOBS": "64",
                },
            ):
                compile_result = subprocess.run(
                    compile_cmd,
                    timeout=180,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    cwd=tmpdir,
                )
            compile_log = compile_result.stdout.decode()
            so_files = glob.glob(f"{tmpdir}/build/**/*.so")
            assert len(so_files) == 1, f"should generate 1 .so file, got {so_files}"
            with open(so_files[0], "rb") as fin:
                bin_content = fin.read()
            ret["ext_filename"] = os.path.basename(so_files[0])
            ret["ext_content"] = bin_content
            ret["msg"] = "compile success"
            success = True
        except subprocess.TimeoutExpired as e:
            success = False
            ret["msg"] = "failed: compilation timed out"
        except Exception as e:
            success = False
            ret["msg"] = f"failed: compilation error: [{e}] log: [{compile_log}]"
        return success, ret


def _compile_ext_optimize(cuda_code_optimize: str) -> Tuple[bool, Dict]:
    ret = {
        "ext_filename": None,
        "ext_content": None,
        "msg": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "model_new_optimize.py"), "w") as fout:
            fout.write(cuda_code_optimize)

        compile_log = ""
        success = True
        try:
            compile_cmd = f"python3 model_new_optimize.py"
            with patch.dict(
                os.environ,
                {
                    "TORCH_CUDA_ARCH_LIST": "8.0",
                    "TORCH_EXTENSIONS_DIR": "build",
                    "MAX_JOBS": "64",
                },
            ):
                compile_result = subprocess.run(
                    compile_cmd,
                    timeout=180,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    cwd=tmpdir,
                )
            compile_log = compile_result.stdout.decode()
            so_files = glob.glob(f"{tmpdir}/build/**/*.so")
            assert len(so_files) == 1, f"should generate 1 .so file, got {so_files}"
            with open(so_files[0], "rb") as fin:
                bin_content = fin.read()
            ret["ext_filename"] = os.path.basename(so_files[0])
            ret["ext_content"] = bin_content
            ret["msg"] = "compile success"
            success = True
        except subprocess.TimeoutExpired as e:
            success = False
            ret["msg"] = "failed: compilation timed out"
        except Exception as e:
            success = False
            ret["msg"] = f"failed: compilation error: [{e}] log: [{compile_log}]"
        return success, ret


TEST_CODE_TMPL = r'''
import torch
import torch.nn.functional as F
import ast
from pathlib import Path
import sys
from contextlib import contextmanager
import time


def rewrite_cuda_model_code(src_path, dst_path):
    """Replace "op = load_inline" with "import op" to separate compilation and execution"""

    model_src = Path(src_path).read_text()
    tree = ast.parse(model_src)

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and isinstance(call := node.value, ast.Call) and \
            ((isinstance(call.func, ast.Attribute) and call.func.attr == 'load_inline') or (isinstance(call.func, ast.Name) and call.func.id == 'load_inline')):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ext_alias = node.targets[0].id
            for kw in call.keywords:
                if kw.arg == 'name':
                    assert isinstance(kw.value, ast.Constant)
                    ext_name = kw.value.value
                    break
            else:
                raise RuntimeError("Cannot find extension name from model_new.py")
            tree.body[i] = ast.parse(f'import {ext_name} as {ext_alias}').body[0]

    model_src = ast.unparse(tree)
    Path(dst_path).write_text(model_src)

rewrite_cuda_model_code(src_path='model_new.py', dst_path='model_new_patch.py')


from model import Model, get_inputs, get_init_inputs
from model_new_patch import ModelNew

def transform_tensors(tensors, fn):
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k:transform_tensors(v, fn) for k, v in tensor.items()}

        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-1, rtol=1e-1)


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith('_') and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function {F.__name__}.{__name} is not allowed in this context."
                )
            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)

def benchmark_model(model, inputs, num_runs=20):
    # Warm-up
    for _ in range(3):
        _ = model(*inputs)
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(*inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return output, avg_time

init_inputs = get_init_inputs()
if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]
torch_model = Model(*init_inputs).cuda()
cuda_model = ModelNew(*init_inputs).cuda()
cuda_model.load_state_dict(torch_model.state_dict())
torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]
torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

torch.cuda.synchronize()

cuda_outputs, cuda_time = benchmark_model(cuda_model,cuda_inputs)

torch.cuda.synchronize()
for _ in range(3):
    torch_outputs = torch_model(*torch_inputs)
torch.cuda.synchronize()

start_time = time.perf_counter()
for _ in range(20):
    torch.cuda.synchronize()  
    torch_outputs = torch_model(*torch_inputs)
    torch.cuda.synchronize()
torch_time =(time.perf_counter() - start_time)/20
check_equal(cuda_outputs, torch_outputs)

print(f"[性能对比] Original CUDA 时间: {cuda_time:.6f}s, PyTorch 时间: {torch_time:.6f}s, 加速比: {torch_time / cuda_time:.2f}x ")

'''


def _exec_eval(
    ext_filename: str, ext_content: bytes, cuda_code: str, pytorch_module: str
) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ext_filename), "wb") as fout:
            fout.write(ext_content)
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)
        with open(os.path.join(tmpdir, "model.py"), "w") as fout:
            fout.write(pytorch_module)
        with open(os.path.join(tmpdir, "test.py"), "w") as fout:
            fout.write(TEST_CODE_TMPL)

        test_log = ""
        try:
            test_cmd = f"python3 test.py"
            test_result = subprocess.run(
                test_cmd,
                timeout=60,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=tmpdir,
            )
            test_log = test_result.stdout.decode()
            if "[性能对比]" in test_log:
                return True, test_log
            else:
                return False, test_log
        except subprocess.TimeoutExpired as e:
            return False, "failed: test timed out"
        except Exception as e:
            return False, f"failed: test error: [{e}] log: [{test_log}]"

    return True, "test success"


TEST_CODE_TMPL_2 = r'''
import torch
import torch.nn.functional as F
import ast
from pathlib import Path
import sys
from contextlib import contextmanager
import time
# op_dir, = list(Path('build').iterdir())
# sys.path.append(str(op_dir))

def rewrite_cuda_model_code(src_path, dst_path):
    """Replace "op = load_inline" with "import op" to separate compilation and execution"""

    model_src = Path(src_path).read_text()
    tree = ast.parse(model_src)

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and isinstance(call := node.value, ast.Call) and \
            ((isinstance(call.func, ast.Attribute) and call.func.attr == 'load_inline') or (isinstance(call.func, ast.Name) and call.func.id == 'load_inline')):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ext_alias = node.targets[0].id
            for kw in call.keywords:
                if kw.arg == 'name':
                    assert isinstance(kw.value, ast.Constant)
                    ext_name = kw.value.value
                    break
            else:
                raise RuntimeError("Cannot find extension name from model_new.py")
            tree.body[i] = ast.parse(f'import {ext_name} as {ext_alias}').body[0]

    model_src = ast.unparse(tree)
    Path(dst_path).write_text(model_src)

rewrite_cuda_model_code(src_path='model_new.py', dst_path='model_new_patch.py')


from model import Model, get_inputs, get_init_inputs
from model_new_patch import ModelNew

def transform_tensors(tensors, fn):
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k:transform_tensors(v, fn) for k, v in tensor.items()}

        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-1, rtol=1e-1)


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith('_') and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function {F.__name__}.{__name} is not allowed in this context."
                )
            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


init_inputs = get_init_inputs()
if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]
torch_model = Model(*init_inputs).cuda()
cuda_model = ModelNew(*init_inputs).cuda()
cuda_model.load_state_dict(torch_model.state_dict())

torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]
torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

for _ in range(5):  # warm-up
    cuda_outputs = cuda_model(*cuda_inputs)


'''


def _exec_eval_ncu(
    ext_filename: str, ext_content: bytes, cuda_code: str, pytorch_module: str
) -> Tuple[bool, str]:
    """Compile and execute test code which checks output with cuda implementation and pytorch module
    :param ext_filename: the cuda extension filename, in the format as "cuda_module.cpython-xxx.so"
    :param ext_content: file content of the extension file
    :param cuda_code: original file content of the python file containing inline cuda code
    :param cuda_code_optimize: optimized file content of the python file containing inline cuda code
    :param pytorch_module: pytorch baseline implementation. Should have Model.forward(...) and get_inputs() api
    :return (status,msg): (True,stdout) for success, (False,stderr) for error
    """
    kernel_name = extract_cuda_kernel_names(cuda_code)
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ext_filename), "wb") as fout:
            fout.write(ext_content)
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)
        with open(os.path.join(tmpdir, "model.py"), "w") as fout:
            fout.write(pytorch_module)
        with open(os.path.join(tmpdir, "test.py"), "w") as fout:
            fout.write(TEST_CODE_TMPL_2)
        with open(os.path.join(tmpdir, "run.sh"), "w") as fout:
            fout.write(
                r"""
METRICS="sm__cycles_active.avg,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_blocks,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__registers_per_thread,\
sm__inst_executed.sum,\
sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum.per_second,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct,\
lts__throughput.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct,\
smsp__sass_average_branch_targets_threads_uniform.pct"
"""
            )
            fout.write(
                f"""

ncu \\
  --kernel-name "{kernel_name}" \\
  --metrics $METRICS \\
  --target-processes all \\
  -o {kernel_name}_report \\
  python test.py 

ncu --import {kernel_name}_report.ncu-rep --csv > record.txt

python extract.py

"""
            )
        with open(os.path.join(tmpdir, "extract.py"), "w") as fout:
            fout.write(
                """
import csv

def extract_metrics(csv_path, target_run="4"):
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # step invalid line
            if not row or row[0] != target_run:
                continue

            # metric (name、metric_unit、value)
            metric_name = row[12]
            metric_unit = row[13]
            metric_value = row[14]
            results.append((metric_name, metric_unit, metric_value))
    
    return results

# Example: outpue extracted result
if __name__ == "__main__":
    path = "record.txt" 
    metrics = extract_metrics(path)
    for name, unit, val in metrics:
        print(f"{name}: {val} {unit}")
"""
            )
        test_log = ""
        try:
            test_cmd = f"bash run.sh"
            test_result = subprocess.run(
                test_cmd,
                timeout=60,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=tmpdir,
            )
            test_log = test_result.stdout.decode()
            return True, test_log
        except subprocess.TimeoutExpired as e:
            return False, "failed: test timed out"
        except Exception as e:
            return False, f"failed: test error: [{e}] log: [{test_log}]"

    return True, "test success"
