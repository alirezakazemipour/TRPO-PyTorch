from .model import CNNModel
import torch
from torch import from_numpy
import numpy as np
from common import explained_variance, categorical_kl, get_flat_params_from, set_flat_params_to
from .conjugate_gradients import cg
from torch.optim.adam import Adam


class Brain:
    def __init__(self, **config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = CNNModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.mse_loss = torch.nn.MSELoss()
        self.value_optimizer = Adam(self.model.critic.parameters(), 1e-4)

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).to(self.device)
        with torch.no_grad():
            dist, value, prob = self.model(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), value.cpu().numpy().squeeze(), log_prob.cpu().numpy(), prob.cpu().numpy()

    def choose_mini_batch(self, states, returns):

        indices = np.random.randint(0, len(states), (self.config["batch_size"] // self.config["value_mini_batch_size"],
                                                     self.config["value_mini_batch_size"]
                                                     )
                                    )
        for idx in indices:
            yield states[idx], returns[idx]

    def train(self, states, actions, rewards, dones, log_probs, probs, values, next_values):
        returns = self.get_returns(rewards, next_values, dones, n=self.config["n_workers"])
        values = np.hstack(values)
        advs = returns - values

        states = from_numpy(states).to(self.device)
        actions = from_numpy(actions).to(self.device)
        advs = from_numpy(advs).to(self.device)
        values_target = from_numpy(returns).to(self.device)
        old_log_prob = from_numpy(log_probs).to(self.device)
        old_probs = from_numpy(probs).to(self.device)

        def get_actor_loss() -> dict:
            dist, _, probs = self.model(states)
            ent = dist.entropy().mean()
            log_prob = dist.log_prob(actions)
            a_loss = -torch.mean((log_prob - old_log_prob).exp() * advs) - self.config["ent_coeff"] * ent
            kl = categorical_kl(probs, old_probs).mean()
            return dict(a_loss=a_loss, ent=ent, kl=kl)

        loss_dict = get_actor_loss()
        self.optimize_actor(loss_dict, get_actor_loss)

        for epoch in range(self.config["value_opt_epoch"]):
            for state, return_ in self.choose_mini_batch(states=states,
                                                         returns=values_target,
                                                         ):
                _, values_pred, _ = self.model(state)
                c_loss = self.mse_loss(return_, values_pred.squeeze(-1))
                self.value_optimizer.zero_grad()
                c_loss.backward()
                self.value_optimizer.step()

        return loss_dict["a_loss"].item() + self.config["ent_coeff"] * loss_dict["ent"].item(), \
               c_loss.item(), \
               loss_dict["ent"].item(), \
               loss_dict["kl"].item(), \
               explained_variance(values, returns)

    def optimize_actor(self, loss_dict: dict, loss_fn):
        loss = loss_dict["a_loss"]
        grads = torch.autograd.grad(loss, self.model.actor.parameters())
        j = torch.cat([g.view(-1) for g in grads]).data

        def fisher_vector_product(y):
            kl = loss_fn()["kl"]
            grads = torch.autograd.grad(kl, self.model.actor.parameters(), create_graph=True)
            flat_grads = torch.cat([g.view(-1) for g in grads])

            inner_prod = (flat_grads * y).sum()
            grads = torch.autograd.grad(inner_prod, self.model.actor.parameters())
            flat_grads = torch.cat([g.reshape(-1) for g in grads]).data
            return flat_grads + y * self.config["damping"]

        opt_dir = cg(fisher_vector_product, -j, self.config["k"])
        quadratic_term = (opt_dir * fisher_vector_product(opt_dir)).sum()
        beta = torch.sqrt(2 * self.config["trust_region"] / quadratic_term)
        opt_step = beta * opt_dir

        with torch.no_grad():
            old_loss = loss_fn()["a_loss"]
            flat_params = get_flat_params_from(self.model.actor)
            exponent_shrink = 1
            params_updated = False
            for _ in range(self.config["line_search_num"]):
                new_params = flat_params + opt_step * exponent_shrink
                set_flat_params_to(new_params, self.model.actor)
                tmp = loss_fn()
                new_loss = tmp["a_loss"]
                new_kl = tmp["kl"]
                improvement = old_loss - new_loss
                if new_kl < 1.5 * self.config["trust_region"] and improvement >= 0 and torch.isfinite(new_loss):
                    params_updated = True
                    break
                exponent_shrink *= 0.5
            if not params_updated:
                set_flat_params_to(flat_params, self.model.actor)

    def get_returns(self, rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, n: int) -> np.ndarray:
        if next_values.shape == ():
            next_values = next_values[None]

        returns = [[] for _ in range(n)]
        for worker in range(n):
            R = next_values[worker]  # noqa
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.config["gamma"] * R * (1 - dones[worker][step])  # noqa
                returns[worker].insert(0, R)

        return np.hstack(returns).astype("float32")

    def set_from_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["policy_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def prepare_to_play(self):
        self.model.eval()
