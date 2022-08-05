from typing import Callable
import torch


def cg(A: Callable, b: torch.Tensor, steps: int, tol: float = 1e-6) -> torch.Tensor: # noqa
    x = torch.zeros_like(b)
    r = b - A(x)
    d = r.clone()
    tol_new = r.t() @ r
    for _ in range(steps):
        if tol_new < tol:
            break
        q = A(d)
        alpha = tol_new / (d.t() @ q)
        x += alpha * d
        r -= alpha * q
        tol_old = tol_new.clone()
        tol_new = r.t() @ r
        beta = tol_new / tol_old
        d = r + beta * d
    return x
