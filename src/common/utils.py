from __future__ import annotations

import time
import torch


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def reset(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
