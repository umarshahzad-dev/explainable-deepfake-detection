"""Experiment-only unittest suite."""
import os
os.environ["DEEPFAKE_EXPERIMENT_TEST_MODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
