import unittest
from typing import List
from llmonk.evaluate.minif2f import check_completions
from lean_dojo import Theorem,  LeanGitRepo
from llmonk.tests.minif2f_examples import GT_PROOFS
import random

def perturb_line(proof: str) -> str:
    """
    Perturb a random line in the proof
    """
    lines = proof.split("\n")
    line = random.randint(0, len(lines)-1)
    lines[line] = lines[line] + "aars"
    return "\n".join(lines)


MUTATIONS = [
    perturb_line
]


def get_mutations(proof: str) -> List[str]:
    """
    Add three random false mutations to the proof
    """
    proofs = [proof]
    corrects = [True] + [False for _ in range(3)]
    for i in range(3):
        proofs.append(
            random.choice(MUTATIONS)(proof)
        )
    return proofs,corrects


class TestCheckCompletions(unittest.TestCase):
    def setUp(self):
        repo = "https://github.com/rah4927/lean-dojo-mew"
        file_path = "MiniF2F/Test.lean"

        repo = LeanGitRepo(repo, "d00c776260c77de7e70125ef0cd119de6c0ff1de")

        self.theorems = [
            Theorem(repo, file_path, "mathd_algebra_478"),
            Theorem(repo, file_path, "mathd_numbertheory_175"),
            Theorem(repo, file_path, "mathd_algebra_314"),
            Theorem(repo, file_path, "mathd_algebra_513"),
            Theorem(repo, file_path, "mathd_algebra_33"),
            Theorem(repo, file_path, "mathd_algebra_137"),
        ]

        self.completions = [
            [
                """norm_num
  rw [h₁, h₂, h₃]
  norm_num
)
```""",
""" rw [h₂, h₃] at h₁
  norm_num at h₁
  linarith
)""",
"""rw [h₂, h₃] at h₁
  field_simp at h₁
  norm_num at h₁
  exact h₁
)
""",
"""  rw [h₂, h₃, h₁]
  norm_num
)
"""
            ],
            [
                """have h : 2^2010 ≡ 4 [MOD 10],
  { apply pow_mod_pow _ 2,
    norm_num,
    exact dec_trivial },
  rw [← Nat.mod_eq_iff] at h,
  exact h
)
```""",
"""  have h₀ : (2^2010) % 10 = ((2^2)^1005) % 10; rfl
  have h₁ : ((2^2)^1005) % 10 = (4^1005) % 10; rfl
  have h₂ : (4^1005) % 10 = 4; norm_num
  linarith
)
```"""
            ],
            [
                "norm_num"
            ],
            [
                "constructor <;> linarith"
            ],
            [
                "field_simp\nnlinarith",    # from official repo
                "norm_num", # incorrect after correct one to check resetting state
                "arstar",
                "field_simp\nnlinarith\narst", # check that it ignores everything after correct proof completion
                ""
            ],
            [
                "have h₁ : ↑x = (575 : ℝ);linarith\nassumption_mod_cast",
                "arstr"
            ]
        ]

        self.expected_results = [
            [True, True, False, False],
            [False, True],
            [True],
            [True],
            [True, False, False, True, False],
            [True, False],
        ]

        for theorem,proof in GT_PROOFS.items():
            self.theorems.append(
                Theorem(repo, file_path, theorem)
            )
            proofs,corrects = get_mutations(proof)
            self.completions.append(proofs)
            self.expected_results.append(corrects)

    def test_check_completions(self):
        for theorem, completion_list, expected in zip(self.theorems, self.completions, self.expected_results):
            with self.subTest(theorem=theorem, completions=completion_list):
                corrects, _, _ = check_completions(theorem, completion_list)
                self.assertEqual(corrects, expected)


if __name__ == '__main__':
    unittest.main()
