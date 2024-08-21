import unittest
from datasets import load_dataset
from tqdm import tqdm
from llmonk.evaluate.code_contests_utils import execution_server_client
from llmonk.evaluate.code_contests import extract_first_code
import random


class TestCheckCompletions(unittest.TestCase):
    def setUp(self):

        dataset = load_dataset("deepmind/code_contests")

        random.seed(0)
        SOLUTIONS_PER_PROBLEM = 5   # make testing quicker
        self.solutions, self.input_expected_output_pairs, self.expected_corrects, self.problems = (
            [],
            [],
            [],
            []
        )
        models = ["Llama-3-70B-Instruct", "Llama-3-8B-Instruct", "Gemma-7B", "Gemma-2B"]
        for model in models:
            sample_dataset = load_dataset(
                "ScalingIntelligence/monkey_business",
                f"CodeContests_{model}"
            )
            for problem in sample_dataset["test"]:
                source_problem = dataset["test"][problem["orig_dset_idx"]]
                
                input_expected_output_pairs = list(
                    zip(
                        source_problem["private_tests"]["input"]
                        + source_problem["public_tests"]["input"]
                        + source_problem["generated_tests"]["input"],
                        source_problem["private_tests"]["output"]
                        + source_problem["public_tests"]["output"]
                        + source_problem["generated_tests"]["output"],
                    )
                )

                selected_samples = random.sample(problem["samples"], min(SOLUTIONS_PER_PROBLEM, len(problem["samples"])))
                selected_is_corrects = [problem["is_corrects"][problem["samples"].index(sample)] for sample in selected_samples]
                self.solutions.append(selected_samples)
                self.input_expected_output_pairs.append(input_expected_output_pairs)
                self.expected_corrects.append(selected_is_corrects)
                self.problems.append(source_problem)

    def test_check_completions(self):
        with execution_server_client.ExecutionServerClient() as client:
            for solutions, input_expected_output_pairs, expected_corrects, problem in tqdm(
                zip(
                    self.solutions,
                    self.input_expected_output_pairs,
                    self.expected_corrects,
                    self.problems
                )
            ):
                corrects = []
                
                for i,code in tqdm(enumerate(solutions)):
                    extracted_code = extract_first_code(code)

                    if extracted_code is None:
                        is_correct = False
                    else:
                        is_correct = client.execute_code(
                            extracted_code,
                            input_expected_output_pairs,
                            timeout=15, # high timeout
                            memory_limit_bytes=2_000_000_000_000,
                        )
                    corrects.append(is_correct)

                    self.assertEqual(is_correct, expected_corrects[i])


if __name__ == "__main__":
    unittest.main()
