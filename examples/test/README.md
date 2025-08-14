# Running Experiments

This directory contains scripts to launch and train various examples to test convergence
and generate reports. The experiments can be run with the script `run_experiments.py`.
This script requires the `mode` and `gpus` to be specified. There are 3 run modes possible:

- `--mode=full`, This will run all examples to completion.
- `--mode=single_step`, This will run all examples for 100 iterations.
- `--mode=unit_tests`, This will run a select set of examples to check.

For example, the command to run all of the unit tests on gpus with ID 0 and 1 is.

```bash
python run_experiments.py --mode=unit_tests --gpus=0,1
```

NOTE: Before running experiments, install QuadPy and related utilities:
```bash
pip install quadpy GPUtil gdown
```
This library is not included in the Docker image for technical reasons.

## Checking Convergence

Convergence can be checked for the `unit_tests` experiments by running the script,

```bash
python run_ci_tests.py
```

This will generate plots in the created folder `./checks` for inspecting convergence.
It will also print convergence information and run times on screen.
