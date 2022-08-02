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
        key = input("Please enter your wandb api key then press enter (just hit the enter if you don't have key):")
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
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary
