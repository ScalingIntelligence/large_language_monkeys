
# https://medium.com/@ingridwickstevens/more-llm-acronyms-an-explainer-on-llama-3s-performance-benchmark-values-36722c6dcabb
# https://arxiv.org/abs/2206.14858
# https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/minerva_math/utils.py#L259
MATH_COT_PROMPT = """Problem:
Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."""


# Examples taken from the MiniF2F training set
# https://github.com/rah4927/lean-dojo-mew/blob/main/MiniF2F/Validation.lean
MINIF2F_FEW_SHOT_PROMPT = """```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem1
  (k x: ℝ)
  (h₀ : x = (13 - Real.sqrt 131) / 4)
  (h₁ : 2 * x^2 - 13 * x + k = 0) :
  k = 19/4 :=
by (
  rw [h₀] at h₁
  rw [eq_comm.mp (add_eq_zero_iff_neg_eq.mp h₁)]
  norm_num
  rw [pow_two]
  rw [mul_sub]
  rw [sub_mul, sub_mul]
  rw [Real.mul_self_sqrt _]
  ring
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem2
  (x p : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p) :
  x - p = 2 - 2 * p :=
by (
  suffices abs (x - 2) = -(x - 2) by
    rw [h₁] at this
    linarith
  apply abs_of_neg
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem3
  (x : ℝ)
  (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x + 2)
  (h₁ : ∀ x, g x = x^2)
  (h₂ : f (g x) = g (f x)) :
  x = - 1/2 :=
by (
  norm_num
  simp_all [-one_div]
  field_simp [h₁]
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem4
  (a b : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a ≠ 2 * b)
  (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5) :
  (a + 11 * b) / (a - b) = 2 :=
by (
  rw [eq_comm]
  refine' (eq_div_iff _).mpr _
  exact sub_ne_zero_of_ne h₀
  rw [eq_comm] at h₂
  suffices : a = 13 * b; linarith
  have key : 5 * (a - 2 * b) = 4 * a + 3 * b; rwa [(eq_div_iff (sub_ne_zero_of_ne h₁)).mp]
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem5
  Int.floor ((9:ℝ) / 160 * 100) = 5 :=
by (
  rw [Int.floor_eq_iff]
  constructor
  all_goals norm_num
)
```

"""