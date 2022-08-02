from .atari_wrappers import make_atari
from multiprocessing import Process


class Worker(Process):
    def __init__(self, id, conn, **config):
        super(Worker, self).__init__()
        self.id = id
        self.config = config
        self.env = None
        self.conn = conn

    def __str__(self):
        return str(self.id)

    def run(self):
        self.env = make_atari(self.config["env_name"], episodic_life=False, seed=self.config["seed"] + self.id)
        print(f"W{self.id}: started.")
        state = self.env.reset(seed=self.config["seed"] + self.id)
        while True:
            self.conn.send(state)
            action = self.conn.recv()
            next_state, reward, done, info = self.env.step(action)
            self.conn.send((next_state, reward, done))
            state = next_state
            if done:
                state = self.env.reset()
