from pathlib import Path
from lean_dojo import (
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    ProofFinished,
    DojoInitError,
    DojoHardTimeoutError,
    DojoCrashError,
)
from typing import List
import re
from tqdm import tqdm
import multiprocessing
import pydra
from copy import deepcopy
import re

from llmonk.utils import (
    get_theorem_name,
    save_yaml,
    EvaluateScriptConfig,
    Timeout,
    load_yaml,
    save_yaml,
)


class ScriptConfig(EvaluateScriptConfig):
    repo_url: str = "https://github.com/rah4927/lean-dojo-mew"
    commit: str = "d00c776260c77de7e70125ef0cd119de6c0ff1de"
    file_path: str = "MiniF2F/Test.lean"


def get_proof_steps(completion: str) -> List[str]:
    """
    Split a lean proof completion into individual proof steps.
    """
    # First, split by newlines
    newline_steps = completion.split("\n")

    steps = []
    for newline_step in newline_steps:
        # Use regex to split on semicolons not surrounded by angle brackets
        # since <;> is a lean operator
        current_steps = re.split(r"(?<!<);(?!>)", newline_step)
        steps.extend(current_steps)

    # Remove Lean indentation and strip whitespace
    steps = [step.replace("·", "").strip() for step in steps]

    return steps


class DojoWrapper:
    def __init__(self, theorem):
        self.theorem = theorem

        self.reset_dojo()

    def reset_dojo(self):
        """
        Reset the dojo environment. This is necesarry whenever there is a timeout to avoid
        corrupted results.
        """
        if hasattr(self, "dojo"):
            self.dojo._cleanup_tmp_dir()
        dojo, init_state = Dojo(self.theorem).__enter__()
        self.dojo = dojo
        self.init_state = init_state


def check_proof(dojo_wrapper, proof_steps):
    state = dojo_wrapper.init_state
    dojo = dojo_wrapper.dojo
    step_cnt = 0

    try:
        for step in proof_steps:
            step_cnt += 1
            state_type = None

            try:
                with Timeout(10):
                    state = dojo.run_tac(state, step)
            except Timeout.Timeout:
                dojo_wrapper.reset_dojo()
                state_type = "Timeout"
                break

            if isinstance(state, ProofFinished) or not isinstance(state, TacticState):
                break

    except (DojoInitError, DojoHardTimeoutError, DojoCrashError) as e:
        state_type = "Exception"

    if isinstance(state, ProofFinished):
        state_type = "Finished"
    else:
        if state_type is None:
            state_type = "TacticError/Incomplete"

    return state_type, step_cnt


def check_completions(theorem: Theorem, completions: List[str]):
    dojo_wrapper = DojoWrapper(theorem)

    corrects, states, num_steps = [], [], []
    for completion in tqdm(completions):
        proof_steps = get_proof_steps(completion)

        state, step_cnt = check_proof(dojo_wrapper, proof_steps)

        correct = state == "Finished"

        corrects.append(correct)
        states.append(state)
        num_steps.append(step_cnt)

    dojo_wrapper.dojo._cleanup_tmp_dir()
    return corrects, states, num_steps


def process_theorem(config: ScriptConfig):

    if config.save_path.exists():
        return

    result = load_yaml(config.sample_path)

    repo = LeanGitRepo(config.repo_url, config.commit)
    theorem = Theorem(repo, config.file_path, get_theorem_name(result["theorem_name"]))

    corrects, _, _ = check_completions(theorem, result["samples"])

    result["is_corrects"] = corrects

    save_yaml(config.save_path, result)


def get_tasks(config):
    sample_paths = Path(config.samples_dir).glob("*.yaml")

    tasks = []
    for sample_path in tqdm(sample_paths, desc="Loading generations"):
        save_path = config.save_dir / sample_path.name

        task_config = deepcopy(config)
        task_config.sample_path = sample_path
        task_config.save_path = save_path

        tasks.append(task_config)

    return tasks


@pydra.main(ScriptConfig)
def main(config):
    tasks = get_tasks(config)
    tasks = sorted(
        tasks, key=lambda x: x.save_path
    )  # sort so the same offset references the same tasks across machines
    tasks = tasks[config.offset : config.limit : config.stride]

    print(f"Evaling on {len(tasks)} problems.")

    if config.num_workers not in [0, None]:
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            _ = list(tqdm(pool.map(process_theorem, tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks):
            process_theorem(task)


if __name__ == "__main__":
    main()