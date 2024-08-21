# All the proofs with answers from https://github.com/rah4927/lean-dojo-mew/blob/main/MiniF2F/Test.lean
GT_PROOFS = {
    "mathd_numbertheory_66": "rfl",

    "mathd_algebra_329": "linarith",

    "mathd_algebra_400": "linarith",

    "mathd_algebra_338": '''have ha : a = -4; linarith
  have hb : b = 2; linarith
  have hc : c = 7; linarith
  rw [ha, hb, hc]
  norm_num''',

    "mathd_numbertheory_254": "norm_num",

    "mathd_numbertheory_728": "norm_num",

    "mathd_algebra_33": '''field_simp
  nlinarith''',

    "mathd_algebra_296": '''rw [abs_of_nonpos]
  norm_num
  norm_num''',

    "mathd_numbertheory_299": "norm_num",

    "mathd_numbertheory_207": "norm_num",

    "mathd_numbertheory_342": "norm_num",

    "mathd_numbertheory_235": "norm_num",

    "mathd_algebra_302": "norm_num",

    "mathd_numbertheory_551": "norm_num",

    "mathd_algebra_304": "norm_num",

    "mathd_algebra_44": "constructor <;> linarith",

    "mathd_algebra_513": "constructor <;> linarith",

    "mathd_algebra_143": '''rw [h₀, h₁]
  norm_num''',

    "mathd_algebra_354": "linarith",

    "mathd_algebra_412": "linarith",
    
    "mathd_algebra_346": '''rw [h₀, h₁]
  norm_num''',

    "mathd_algebra_24": "nlinarith",

    "mathd_algebra_427": '''have h₃ := congr (congr_arg Add.add h₀) h₁
  linarith''',

    "mathd_algebra_142": "linarith",

    "mathd_algebra_270": '''rw [h₀, h₀]
  norm_num
  linarith
  rw [h₀]
  norm_num
  linarith''',

    "mathd_algebra_263": '''revert y h₀ h₁
  intro x hx
  rw [Real.sqrt_eq_iff_sq_eq hx]
  swap
  norm_num
  intro h
  nlinarith''',

    "mathd_algebra_314": '''rw [h₀]
  norm_num'''
}