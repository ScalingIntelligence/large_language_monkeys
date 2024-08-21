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


def get_few_shot_prompt(item):
    few_shot_items = item["few_shot_items"]

    few_shot_pieces = []
    for f in few_shot_items:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/gsm8k/gsm8k.yaml
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)

    few_shot_prompt = "".join(few_shot_pieces)

    return few_shot_prompt


def run_inference(item, config: GenerateScriptConfig):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    few_shot_prompt = get_few_shot_prompt(item)
    prompt = few_shot_prompt + f"Question: {item['question']}\nAnswer:"

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
            "stop": config.stop_strings,
            "logprobs": 1,
        }

        response = requests.post(url, json=body)
        respj = response.json()
        samples.extend(respj["text"])

    out = {
        "prompt": prompt,
        "question": item["question"],
        "samples": samples,
        "gt_answer": item["answer"],
    }

    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(
    config: GenerateScriptConfig,
):

    test_dataset = list(load_dataset("gsm8k", "main", split="test"))
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))

    print(f"Number of test items: {len(test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(config.seed)

    for i, data in enumerate(train_dataset):
        data["id"] = i

    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, config.num_few_shot)
        data["id"] = i
        data["few_shot_items"] = few_shot_items

    random.shuffle(test_dataset)

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(test_dataset)

    if config.stride is not None:
        stride = config.stride
    else:
        stride = 1

    if config.offset is not None:
        offset = config.offset
    else:
        offset = 0

    test_dataset = test_dataset[offset:limit:stride]

    print(f"Total number of items to process: {len(test_dataset)}")

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config)

        if config.num_workers not in [0, None]:
            with multiprocessing.Pool(config.num_workers) as pool:
                predictions = list(
                    tqdm(
                        pool.imap_unordered(go_func, test_dataset),
                        total=len(test_dataset),
                    )
                )
        else:
            predictions = []
            for item in tqdm(test_dataset):
                predictions.append(go_func(item))


if __name__ == "__main__":
    main()
