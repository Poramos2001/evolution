import evogym
import gymnasium as gym
import evogym.envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import ray
import json
from tqdm import tqdm
from evogym.utils import get_full_connectivity
import evolve_tools as et

from evogym import sample_robot
import copy

import cma


# ---------------------- Network ----------------------
class Network(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)

    def reset(self):
        pass

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x) + 0.6
        return x

# ---------------------- Agent ----------------------
class Agent:
    def __init__(self, Net, config, genes=None):
        import gymnasium as gym
        import evogym.envs
        self.config = config
        self.Net = Net
        self.device = torch.device("cpu")  # Always CPU for Ray compatibility
        self.model = None
        self.fitness = None
        self.make_network()
        if genes is not None:
            self.genes = genes

    def make_network(self):
        n_in = self.config["n_in"]
        h_size = self.config["h_size"]
        n_out = self.config["n_out"]
        self.model = self.Net(n_in, h_size, n_out).to(self.device).double()
        return self

    @property
    def genes(self):
        if self.model is None:
            return None
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params):
        if self.model is None:
            self.make_network()
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.fitness = None
        return self

    def act(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy().flatten()
        return actions

# ---------------------- Environment ----------------------
def make_env(env_name, seed=None, robot=None, **kwargs):
    if robot is None:
        env = gym.make(env_name)
    else:
        connections = get_full_connectivity(robot)
        env = gym.make(env_name, body=robot)
    if seed is not None:
        env.seed(seed)
    return env

# ---------------------- Evaluation ----------------------
def evaluate(agent, env, max_steps=500, render=False):
    obs, _ = env.reset()
    agent.model.reset()
    reward = 0
    done = False
    steps = 0
    while not done and steps < max_steps:
        if render:
            env.render()
        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        reward += r
        steps += 1
    return reward

# ---------------------- Ray Remote Evaluation ----------------------
@ray.remote
def mp_eval_remote(genes, config):
    agent = Agent(Network, config)
    agent.genes = genes
    env = make_env(config["env_name"], robot=config["robot"])
    fitness = evaluate(agent, env, max_steps=config["max_steps"])
    env.close()
    return fitness

# ---------------------- Evolution Strategy ----------------------
def get_cfg(env_name, robot, n=None):
    env = make_env(env_name, robot=robot)
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": env.action_space.shape[0],
    }
    if n is not None:
        cfg["h_size"] = n*cfg["n_in"]
    env.close()
    return cfg



def CMA_ES(config):
    ray.init(ignore_reinit_error=True)

    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg.update(config)

    # Init agent
    agent = Agent(Network, cfg)
    theta_init = agent.genes

    es = cma.CMAEvolutionStrategy(theta_init, cfg["sigma"],
                                  {'popsize': cfg["lambda"]})

    fits = []
    total_evals = []
    bar = tqdm(range(cfg["generations"]))

    for gen in bar:
        solutions = es.ask()

        # Ray parallel fitness evaluation
        futures = [mp_eval_remote.remote(genes, cfg) for genes in solutions]
        fitnesses_raw = ray.get(futures)

        # Negate fitness (CMA-ES minimize)
        fitnesses = [-f for f in fitnesses_raw]

        es.tell(solutions, fitnesses)

        best_fit_gen = max(fitnesses_raw)  # raw (positive) fitness
        fits.append(best_fit_gen)
        total_evals.append((gen + 1) * cfg["lambda"])

        bar.set_description(f"Best gen fit: {best_fit_gen}")

        # Track best ever
        if best_fit_gen > getattr(agent, "best_fit", -np.inf):
            best_idx = np.argmax(fitnesses_raw)
            agent.best_fit = best_fit_gen
            agent.best_genes = solutions[best_idx].copy()

    ray.shutdown()

    # Set final best
    agent.genes = agent.best_genes
    agent.fitness = agent.best_fit

    # Plot
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("CMA-ES Progress")
    plt.show()

    return agent

# ---------------------- Saving and Loading ----------------------
def save_solution(a, cfg, name="solution.json"):
    save_cfg = {}
    for i in ["env_name", "robot", "n_in", "h_size", "n_out"]:
        assert i in cfg, f"{i} not in config"
        save_cfg[i] = cfg[i]
    save_cfg["robot"] = cfg["robot"].tolist()
    save_cfg["genes"] = a.genes.tolist()
    save_cfg["fitness"] = float(a.fitness)
    # save
    with open(name, "w") as f:
        json.dump(save_cfg, f)
    return save_cfg

def load_solution(name="solution.json"):
    with open(name, "r") as f:
        cfg = json.load(f)
    cfg["robot"] = np.array(cfg["robot"])
    cfg["genes"] = np.array(cfg["genes"])
    a = Agent(Network, cfg, genes=cfg["genes"])
    a.fitness = cfg["fitness"]
    return a

# ---------------------- Main ----------------------
if __name__ == "__main__":
    climber = np.array([
        [3, 3, 4, 3, 3],
        [0, 4, 4, 4, 0],
        [0, 2, 4, 4, 0],
        [1, 4, 1, 2, 1],
        [3, 3, 1, 3, 3]
        ])

    config = {
        "env_name": "Climber-v2",
        "robot": climber,
        "generations": 500,
        "lambda": 30, # Population size
        "sigma": 2,
        "mu": 5, # Parents pop size
        "lr": 1, # Learning rate
        "max_steps": 100, # to change to 500
        "plot_name": "climber.png",
        "plot":True,
        "n": 3 # The network h_size = n*n_in, None sets h_size to 32
    }

    a = CMA_ES(config)
    env = et.make_env(config["env_name"], robot=config["robot"],
                    render_mode="rgb_array")
    et.generate_gif(gif_name=f"climber.gif", a=a, env=env)

    cfg = et.get_cfg(config["env_name"], robot=config["robot"],
                    n=config["n"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    et.save_solution(a, cfg, name=f"climber.json")

