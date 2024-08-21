import unittest
from datasets import load_dataset
from tqdm import tqdm
from llmonk.evaluate.math_datasets import is_correct as evaluate_correctness
import random


class TestCheckCompletions(unittest.TestCase):
    def setUp(self):
        datasets = ["GSM8K", "MATH"]
        models = ["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"]
        self.gt_answers = {dataset: [] for dataset in datasets}
        self.samples = {dataset: [] for dataset in datasets}
        self.is_corrects = {dataset: [] for dataset in datasets}
        for dataset in datasets:
            for model in models:
                sample_dataset = load_dataset("ScalingIntelligence/monkey_business", f"{dataset}_{model}", download_mode="force_redownload")
                for problem in sample_dataset["test"]:
                    gt_answer = problem["gt_answer"]
                    for sample, is_correct in zip(problem["samples"], problem["is_corrects"]):
                        self.gt_answers[dataset].append(gt_answer)
                        self.samples[dataset].append(sample)
                        self.is_corrects[dataset].append(is_correct)
            
            # take 1000 randomly for faster eval
            random.seed(0)
            indices = random.sample(range(len(self.gt_answers[dataset])), 1000)
            self.gt_answers[dataset] = [self.gt_answers[dataset][i] for i in indices]
            self.samples[dataset] = [self.samples[dataset][i] for i in indices]
            self.is_corrects[dataset] = [self.is_corrects[dataset][i] for i in indices]
        
    def test_check_completions(self):
        for dataset in self.gt_answers.keys():
            gt_answers = self.gt_answers[dataset]
            samples = self.samples[dataset]
            is_corrects = self.is_corrects[dataset]
            for gt_answer,sample,is_correct in tqdm(zip(gt_answers,samples,is_corrects)):
                self.assertEqual(evaluate_correctness(sample, gt_answer, dataset.lower()), is_correct)


if __name__ == '__main__':
    unittest.main()
