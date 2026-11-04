import random
import numpy as np


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    import torch
    np_state = np.random.get_state()
    safe_numpy = (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4])
    state = {"python": random.getstate(), "numpy": safe_numpy, "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), values[2], values[3], values[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])
