import os
import numpy as np
import random
import torch
import wandb


def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def init_wandb(online_mode=False):
    if os.path.exists("api_key.wandb"):
        with open("api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            if not online_mode:
                os.environ["WANDB_MODE"] = "offline"
    else:
        if not online_mode:
            os.environ["WANDB_MODE"] = "offline"
        key = input("Please enter your wandb api key then press enter (just hit the enter if you don't have any):")
        wandb.login(key=key)


# Calculates if value function is a good predictor of the returns (ev > 1)
# or if it's just worse than predicting nothing (ev =< 0)
def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y) + 1e-6
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def categorical_kl(p_nk: torch.Tensor, q_nk: torch.Tensor):
    # https://github.com/joschu/modular_rl/blob/master/modular_rl/distributions.py
    ratio_nk = p_nk / (q_nk + 1e-6)
    ratio_nk[p_nk == 0] = 1
    ratio_nk[(q_nk == 0) & (p_nk != 0)] = np.inf
    return (p_nk * torch.log(ratio_nk)).sum(dim=1)


def get_flat_params_from(model: torch.nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params_to(params: torch.nn.Module.parameters, model: torch.nn.Module):
    pointer = 0
    for p in model.parameters():
        p.data.copy_(params[pointer:pointer + p.data.numel()].view_as(p.data))
        pointer += p.data.numel()
