import argparse
import os
import warnings


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--env_name", default="PongNoFrameskip-v4", type=str, help="Name of the environment.")
    parser.add_argument("--num_worker", default=-1, type=int,
                        help="Number of parallel workers. (-1) to use as many as cpu cores.")
    parser.add_argument("--total_iterations", default=800000, type=int, help="The total number of iterations.")
    parser.add_argument("--interval", default=1500, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by iterations.")
    parser.add_argument("--online_wandb", action="store_true", help="Run wandb in online mode.")
    parser.add_argument("--do_test", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--render", action="store_true",
                        help="The flag determines whether to render each agent or not.")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="The flag determines whether to train from scratch or continue previous tries.")
    parser.add_argument("--seed", default=132, type=int, help="The random seed.")

    parser_params = parser.parse_args()

    # region default parameters
    default_params = {"state_shape": (4, 84, 84),
                      "lr": 0.25,
                      "batch_size": 640,
                      "gamma": 0.99,
                      "ent_coeff": 0.01,  # noqa
                      "critic_coeff": 0.5,  # noqa
                      "n_workers": os.cpu_count() if parser_params.num_worker == -1 else parser_params.num_worker,
                      }

    # endregion
    total_params = {**vars(parser_params), **default_params}
    if default_params["n_workers"] > os.cpu_count():
        warnings.warn("🚨You're using more workers than your machine's CPU cores!🚨")
    return total_params
