import torch
from datasets import load_dataset
from tqdm import tqdm
import pydra
import multiprocessing
import requests
import re
from functools import partial

from llmonk.utils import (
    save_yaml,
    GenerateScriptConfig,
)
from llmonk.generate.vllm_utils import vllm_manager
from llmonk.generate.prompts import MINIF2F_FEW_SHOT_PROMPT


def replace_theorem_name(lean_code, new_name):
    """
    Replace dataset's theorem name with a generic name
    to avoid leaking information about how to solve the problem
    """
    pattern = r"theorem\s+\w+\s*\n"
    replacement = f"theorem{new_name}\n"
    modified_code = re.sub(pattern, replacement, lean_code)
    return modified_code


def get_lean_prompt(data, theorem_name: str, add_solution: bool = False):
    header = "Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.\n"
    header += "```" + data["header"]
    stmt = data["formal_statement"].replace(" sorry", "").replace("sorry", "")
    if add_solution:
        prompt = header + "\n" + stmt + data["solution"] + "```"
    else:
        prompt = header + "\n" + stmt + "\nby (\n"

    prompt = replace_theorem_name(prompt, theorem_name)

    return prompt


def run_inference(item, config: GenerateScriptConfig):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    # we use five few-shot examples
    prompt = MINIF2F_FEW_SHOT_PROMPT + get_lean_prompt(item, theorem_name="6")

    url = f"http://localhost:{config.vllm_port}/generate"

    num_samples = config.num_samples
    batch_size = config.batch_size

    assert num_samples % batch_size == 0

    samples = []
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):

        body = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "n": batch_size,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "include_stop_str_in_output": True,
            "stop": config.stop_strings,
        }

        response = requests.post(url, json=body)
        respj = response.json()
        samples.extend(respj["text"])

    out = {
        "prompt": prompt,
        "question": item["formal_statement"],
        "samples": samples,
        "theorem_name": item["id"],
    }

    save_yaml(outpath, out)


@torch.no_grad()
@pydra.main(GenerateScriptConfig)
def main(
    config: GenerateScriptConfig,
):
    dataset = load_dataset("cat-searcher/minif2f-lean4")
    math_problems = [p for p in dataset["test"] if "mathd" in p["id"]]

    assert len(math_problems) == 130

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(math_problems)

    if config.stride is not None:
        stride = config.stride
    else:
        stride = 1

    if config.offset is not None:
        offset = config.offset
    else:
        offset = 0

    math_problems = math_problems[offset:limit:stride]

    print(f"Total number of items to process: {len(math_problems)}")

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config)

        if config.num_workers not in [0, None]:
            with multiprocessing.Pool(config.num_workers) as pool:
                predictions = list(
                    tqdm(
                        pool.imap_unordered(go_func, math_problems),
                        total=len(math_problems),
                    )
                )
        else:
            predictions = []
            for item in tqdm(math_problems):
                predictions.append(go_func(item))


if __name__ == "__main__":
    main()
