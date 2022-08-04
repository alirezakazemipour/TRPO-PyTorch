import time
import numpy as np
import os
import datetime
import glob
from collections import deque
import psutil
import sys
import math
import wandb
import torch


class Logger:
    def __init__(self, brain, **config):
        self.config = config
        self.brain = brain
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_reward = 0
        self.running_reward = 0
        self.max_episode_rewards = -np.inf
        self.episode_length = 0
        self.moving_avg_window = 10
        self.running_training_logs = 0
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self.running_last_10_r = 0  # It is not correct but does not matter.
        self.to_gb = lambda x: x / 1024 / 1024 / 1024

        sys.stdout.write("\033[;1m")  # bold code
        print("params:", self.config)
        sys.stdout.write("\033[0;0m")  # Reset code

        # wandb.watch(agent.online_model)
        if not self.config["do_test"] and self.config["train_from_scratch"]:
            self.create_wights_folder(self.log_dir)
            wandb.init(project="TRPO",  # noqa
                       config=config,
                       job_type="train",
                       name=self.log_dir
                       )

        self.exp_avg = lambda x, y: 0.99 * x + 0.01 * y if (y != 0).all() else y

    @staticmethod
    def create_wights_folder(dir):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        os.mkdir("weights/" + dir)

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_iteration(self, *args):
        iteration, training_logs = args

        if np.isnan(np.mean(training_logs[:-1])):
            raise RuntimeError(f"NN has output NaNs! {training_logs}")
        if math.isnan(training_logs[-1]):
            training_logs = list(training_logs[:-1])
            training_logs.append(self.running_training_logs[-1])

        self.running_training_logs = self.exp_avg(self.running_training_logs, np.array(training_logs))

        if iteration % (self.config["interval"] // 3 + 1) == 0:
            self.save_params(self.episode, iteration)

        metrics = {"Running Episode Reward": self.running_reward,
                   "Running last 10 Reward": self.running_last_10_r,
                   "Max Episode Reward": self.max_episode_rewards,
                   "Episode Length": self.episode_length,
                   "Running Actor Loss": self.running_training_logs[0],
                   "Running Critic Loss": self.running_training_logs[1],
                   "Running Entropy": self.running_training_logs[2],
                   "Running Explained variance": self.running_training_logs[3],
                   "episode": self.episode,
                   "iteration": iteration
                   }
        wandb.log(metrics)

        self.off()
        if iteration % self.config["interval"] == 0:
            ram = psutil.virtual_memory()
            print("\nIter: {}| "
                  "E: {}| "
                  "E_Reward: {:.1f}| "
                  "E_Running_Reward: {:.1f}| "
                  "Iter_Duration: {:.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time: {} "
                  .format(iteration,
                          self.episode,
                          self.episode_reward,
                          self.running_reward,
                          self.duration,
                          self.to_gb(ram.used),
                          self.to_gb(ram.total),
                          datetime.datetime.now().strftime("%H:%M:%S"),
                          )
                  )
        self.on()

    def log_episode(self, *args):
        self.episode, self.episode_reward, episode_length = args

        self.max_episode_rewards = max(self.max_episode_rewards, self.episode_reward)

        if self.episode == 1:
            self.running_reward = self.episode_reward
            self.episode_length = episode_length
        else:
            self.running_reward = self.exp_avg(self.running_reward, self.episode_reward)
            self.episode_length = 0.99 * self.episode_length + 0.01 * episode_length

        self.last_10_ep_rewards.append(self.episode_reward)
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.running_last_10_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')

    # region save_params
    def save_params(self, episode, iteration):
        torch.save({"model_state_dict": self.brain.model.state_dict(),
                    # "optimizer_state_dict": self.brain.optimizer.state_dict(),
                    "iteration": iteration,
                    "episode": episode,
                    "running_reward": self.running_reward,
                    "running_last_10_r": self.running_last_10_r
                    if not isinstance(self.running_last_10_r, np.ndarray) else self.running_last_10_r[0],
                    "running_training_logs": list(self.running_training_logs)
                    },
                   "weights/" + self.log_dir + "/params.pth"
                   )

    # endregion

    # region load_weights
    def load_weights(self):
        model_dir = glob.glob("weights/*")
        model_dir.sort()
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        checkpoint = torch.load("weights/" + self.log_dir + "/params.pth")

        self.brain.model.load_state_dict(checkpoint["model_state_dict"])
        self.brain.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.running_last_10_r = checkpoint["running_last_10_r"]
        self.running_training_logs = np.asarray(checkpoint["running_training_logs"])
        self.running_reward = checkpoint["running_reward"]

        if not self.config["do_test"] and not self.config["train_from_scratch"]:
            wandb.init(project="ACKTR",  # noqa
                       config=self.config,
                       job_type="train",
                       name=self.log_dir
                       )
        return checkpoint["iteration"], checkpoint["episode"]
    # endregion
