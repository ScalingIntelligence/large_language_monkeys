import torch
from datasets import load_dataset
from tqdm import tqdm
import pydra
import multiprocessing
import random
import requests
from functools import partial

from llmonk.utils import save_yaml, GenerateScriptConfig
from llmonk.generate.vllm_utils import vllm_manager

PYTHON3_LANGUAGE_ID = 3


def get_python_solutions(
    item,
    filter_non_ascii: bool = True,
    incorrect_solutions: bool = False,
):
    if incorrect_solutions:
        solution_key = "incorrect_solutions"
    else:
        solution_key = "solutions"

    python_solutions = []
    for i, (solution, lang_id) in enumerate(
        zip(
            item[solution_key]["solution"],
            item[solution_key]["language"],
        )
    ):
        if filter_non_ascii:
            if not solution.isascii():
                continue

        if lang_id == PYTHON3_LANGUAGE_ID:
            python_solutions.append(solution)

    return python_solutions


IMAGE_TAGS = ["<image>", "[Image]"]


def has_image_tags(description):
    for tag in IMAGE_TAGS:
        if tag in description:
            return True
    return False


CODELLAMA_PROMPT = "Q: Write python code to solve the following coding problem that obeys the constraints and passes the example test cases. The output code needs to read from and write to standard IO. Please wrap your code answer using ```:"


def problem_to_prompt(problem, add_solution=True):
    prompt = f"{CODELLAMA_PROMPT}\n{problem['description']}\nA:"
    if add_solution:
        prompt += f" ```{problem['python_solutions'][0].strip()}```"

    return prompt


def get_prompt(item):
    prompt = "\n".join(
        [problem_to_prompt(few_shot_item) for few_shot_item in item["few_shot_items"]]
    )
    prompt += "\n" + problem_to_prompt(item, add_solution=False)
    return prompt


def get_timeout(item):
    timeout_seconds = 0
    if item["time_limit"] is not None:
        timeout_seconds += item["time_limit"]["seconds"]
        timeout_seconds += item["time_limit"]["nanos"] / 1_000_000_000

    if timeout_seconds == 0:
        timeout_seconds = None
    return timeout_seconds


def get_test_cases(item):
    return {
        "input": item["public_tests"]["input"]
        + item["private_tests"]["input"]
        + item["generated_tests"]["input"],
        "output": item["public_tests"]["output"]
        + item["private_tests"]["output"]
        + item["generated_tests"]["output"],
    }


def run_inference(item, config: GenerateScriptConfig):
    outpath = config.save_dir / f"{item['name']}.yaml"
    if outpath.exists():
        return

    prompt = get_prompt(item)
    url = f"http://localhost:{config.vllm_port}/generate"

    num_samples = config.num_samples
    batch_size = config.batch_size

    assert num_samples % batch_size == 0

    samples = []
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['name']}"):

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
        "question": item["description"],
        "samples": samples,
        "test_cases": get_test_cases(item),
        "timeout": get_timeout(item),
    }

    save_yaml(outpath, out)


@torch.no_grad()
@pydra.main(GenerateScriptConfig)
def main(
    config: GenerateScriptConfig,
):
    dataset = load_dataset("deepmind/code_contests")
    few_shot_dataset = [p for p in dataset["train"]]
    test_dataset = [p for p in dataset["test"]]

    random.seed(config.seed)

    few_shot_items_with_solutions = []
    for i, data in enumerate(few_shot_dataset):
        python_solutions = get_python_solutions(data)
        data["python_solutions"] = python_solutions
        if len(python_solutions) > 0 and not has_image_tags(data["description"]):
            few_shot_items_with_solutions.append(data)

    no_image_test_dataset = []
    for i, data in enumerate(test_dataset):
        if has_image_tags(data["description"]):
            continue
        few_shot_items = random.sample(
            few_shot_items_with_solutions, config.num_few_shot
        )
        data["few_shot_items"] = few_shot_items
        no_image_test_dataset.append(data)

    random.shuffle(no_image_test_dataset)

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(no_image_test_dataset)

    if config.stride is not None:
        stride = config.stride
    else:
        stride = 1

    if config.offset is not None:
        offset = config.offset
    else:
        offset = 0

    no_image_test_dataset = no_image_test_dataset[offset:limit:stride]

    print(f"Total number of items to process: {len(no_image_test_dataset)}")

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config)

        if config.num_workers not in [0, None]:
            with multiprocessing.Pool(config.num_workers) as pool:
                predictions = list(
                    tqdm(
                        pool.imap_unordered(go_func, no_image_test_dataset),
                        total=len(no_image_test_dataset),
                    )
                )
        else:
            predictions = []
            for item in tqdm(no_image_test_dataset):
                predictions.append(go_func(item))


if __name__ == "__main__":
    main()
