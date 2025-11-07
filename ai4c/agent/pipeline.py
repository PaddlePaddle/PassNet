import os
import argparse
from openai import OpenAI
from ai4c.agent.prompt_cuda import (
    generate_default_cuda_prompt,
    judge_optimize_prompt,
    judge_correct_prompt,
    coder_optimize_prompt,
    coder_correct_prompt,
)
from ai4c.utils.kernel_utils import (
    extract_cuda_code,
    remove_pybind_module,
    _compile_kernel,
    _exec_eval,
    _exec_eval_ncu,
)

# Unstable API
API_KEY = os.getenv("API_KEY", None)
BASE_URL = os.getenv("BASE_URL", None)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "system",
                "content": "You are a coding assistant that writes CUDA code.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )

    result = response.choices[0].message.content.strip()
    return result


# !Unstable API


def generate_one_cuda_kernel(torch_file_path: str, store_dir: str, max_iters: int):
    with open(torch_file_path, "r", encoding="utf-8") as f:
        torch_model_code = f.read()
    prompt = generate_default_cuda_prompt(torch_model_code)
    response = generate(prompt)

    cuda_code = remove_pybind_module(extract_cuda_code(response))

    def optimize_with_ncu(
        cuda_code: str, torch_code: str, compile_ret: dict, output_file: str
    ):
        _, ncu_metric = _exec_eval_ncu(
            compile_ret["ext_filename"],
            compile_ret["ext_content"],
            cuda_code,
            torch_code,
        )
        prompt = judge_optimize_prompt(torch_code, cuda_code, ncu_metric)
        strategy = generate(prompt)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"OPTIMIZE_STRATGY:\n{{\n {strategy}\n }}\n")
        prompt = coder_optimize_prompt(cuda_code, strategy)
        cuda_code = generate(prompt)
        cuda_code = extract_cuda_code(cuda_code)
        cuda_code = remove_pybind_module(cuda_code)
        return cuda_code

    # iterative optimization
    prompt = None
    for it in range(max_iters):
        print(f"=== Iteration {it} ===", flush=True)

        success_functional = False
        success_compilable = False

        fp_output_file = os.path.join(store_dir, f"{it}.log")
        os.makedirs(os.path.dirname(fp_output_file), exist_ok=True)

        success_compilable, ret_compile = _compile_kernel(cuda_code)
        with open(fp_output_file, "a", encoding="utf-8") as f:
            f.write(f"CUDA_CODE:\n{{\n{cuda_code} \n}}\n")
            f.write(f"COMPILE_RESULT:\n{{\n {ret_compile['msg']}\n }}\n")

        if success_compilable:  # compilable
            success_functional, ret_eval = _exec_eval(
                ret_compile["ext_filename"],
                ret_compile["ext_content"],
                cuda_code,
                torch_model_code,
            )
            with open(fp_output_file, "a", encoding="utf-8") as f:
                f.write(f"EVAL_RESULT:\n{{\n {ret_eval[:2048]}\n }}\n")
            if success_functional:  # functional correct
                cuda_code = optimize_with_ncu(
                    cuda_code, torch_model_code, ret_compile, fp_output_file
                )
            else:  # functional error
                prompt = judge_correct_prompt(
                    ret_eval[:2048], torch_model_code, cuda_code
                )
                modify_text = generate(prompt)
                with open(fp_output_file, "a", encoding="utf-8") as f:
                    f.write(f"CORRECT_MODIFY_ANALYSE:\n{{\n {modify_text}\n }}\n")
                prompt = coder_correct_prompt(ret_eval[:2048], cuda_code, modify_text)
                cuda_code = generate(prompt)
                cuda_code = extract_cuda_code(cuda_code)
                cuda_code = remove_pybind_module(cuda_code)

        else:  # compile error
            prompt = judge_correct_prompt(
                ret_compile["msg"][:2048], torch_model_code, cuda_code
            )
            modify_text = generate(prompt)
            with open(fp_output_file, "a", encoding="utf-8") as f:
                f.write(f"COMPILE_MODIFY_ANALYSE:\n{{\n {modify_text}\n }}\n")
            prompt = coder_correct_prompt(
                ret_compile["msg"][:2048], cuda_code, modify_text
            )
            cuda_code = generate(prompt)
            cuda_code = extract_cuda_code(cuda_code)
            cuda_code = remove_pybind_module(cuda_code)


def main(args):
    # TODO Fetch the subgraph (torch / graph) and its inputs; the subgraph should be a coarse-grained graph (< 32 ops)
    # TODO agent performs fine-grained subgraph partitioning

    # The agent optimizes the kernel (using KernelBench as the benchmark input for testing)
    generate_one_cuda_kernel(args.case_path, args.output_dir, max_iters=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-path",
        type=str,
        required=True,
        help="The path to the test case Torch file (Only support KernelBench Case Now).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="The maximum number of optimization iterations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tmp_agent/",
        help="The directory to store the generated CUDA kernels.",
    )

    args = parser.parse_args()
    main(args=args)
