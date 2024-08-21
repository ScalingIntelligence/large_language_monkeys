# Environment variables

Folder where you want to save samples/evaluations:
```
export SAVE_DIR=SET ME!
```

Folder where you want to save tmp files for MiniF2F-MATH evaluation:
```
export SCRATCH_DIR=SET ME!
mkdir $SCRATCH_DIR
mkdir $SCRATCH_DIR/cache
mkdir $SCRATCH_DIR/lean_tmp
```

# Generate

# GSM8K
```
python llmonk/generate/gsm8k.py model=meta-llama/Meta-Llama-3-8B-Instruct save_dir=$SAVE_DIR/gsm8k_samples --list vllm_args --disable-log-requests list--
```

# MATH
```
python llmonk/generate/MATH.py model=meta-llama/Meta-Llama-3-8B-Instruct save_dir=$SAVE_DIR/math_samples --list vllm_args --disable-log-requests list--
```

## CodeContests
```
python llmonk/generate/code_contests.py model=meta-llama/Meta-Llama-3-8B-Instruct save_dir=$SAVE_DIR/cc_samples --list vllm_args --disable-log-requests list--
```

## MiniF2F
```
python llmonk/generate/minif2f.py model=meta-llama/Meta-Llama-3-8B-Instruct save_dir=$SAVE_DIR/minif2f_samples --list vllm_args --disable-log-requests list--
```

# Evaluate

# GSM8K
```
python llmonk/evaluate/math_datasets.py samples_dir=$SAVE_DIR/gsm8k_samples save_dir=$SAVE_DIR/gsm8k_eval dset=gsm8k
```

# MATH
```
python llmonk/evaluate/math_datasets.py samples_dir=$SAVE_DIR/math_samples save_dir=$SAVE_DIR/math_eval dset=math
```

## CodeContests
Note that the first time you run code evaluation, execute the command from the `llmonk/evaluate/code_contests_utils` directory to build the docker image with the required paths.
```
python llmonk/evaluate/code_contests.py samples_dir=$SAVE_DIR/cc_samples save_dir=$SAVE_DIR/cc_eval
```

## MiniF2F
```
CONTAINER=native CACHE_DIR=$SCRATCH_DIR/cache TMP_DIR=$SCRATCH_DIR/lean_tmp python llmonk/evaluate/minif2f.py samples_dir=$SAVE_DIR/minif2f_samples save_dir=$SAVE_DIR/minif2f_eval
```


# Unit tests

## Math dataset evaluation code
```
python llmonk/tests/math_datasets.py
```

## CodeContests executor
Note that the first time you run code evaluation, execute the command from the `llmonk/evaluate/code_contests_utils` directory to build the docker image with the required paths.
```
python llmonk/tests/code_contests.py
```

## CodeContest equality checks on output
```
python llmonk/tests/compare_results.py
```

## MiniF2F verifier
```
CONTAINER=native CACHE_DIR=$SCRATCH_DIR/cache TMP_DIR=$SCRATCH_DIR/lean_tmp python llmonk/tests/minif2f.py
```

