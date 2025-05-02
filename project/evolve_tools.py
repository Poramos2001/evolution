import imageio
import gymnasium as gym
import evogym.envs
from evogym.utils import get_full_connectivity
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from cmaes import CMA


class Network(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)
 
        self.n_out = n_out

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


class Agent:
    def __init__(self, Net, config, genes = None):
        self.config = config
        self.Net = Net
        self.model = None
        self.fitness = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.make_network()
        if genes is not None:
            self.genes = genes

    def __repr__(self):  # pragma: no cover
        return (f"Agent {self.model} > fitness={self.fitness} with "
                f"{len(self.genes)} parameters")

    def __str__(self):  # pragma: no cover
        return self.__repr__()

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
        assert len(params) == len(
            self.genes), "Genome size does not fit the network size"
        if np.isnan(params).any():
            raise
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.fitness = None
        return self

    def mutate_ga(self):
        genes = self.genes
        n = len(genes)
        f = np.random.choice([False, True], size=n, p=[1/n, 1-1/n])
        
        new_genes = np.empty(n)
        new_genes[f] = genes[f]
        noise = np.random.randn(n-sum(f))
        new_genes[~f] = noise
        return new_genes

    def act(self, obs):
        # continuous actions
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy()
        return actions


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


def make_env(env_name, seed=None, robot=None, **kwargs):
    if robot is None: 
        env = gym.make(env_name, **kwargs)
    else:
        connections = get_full_connectivity(robot)
        env = gym.make(env_name, body=robot, connections=connections, **kwargs)
    env.robot = robot
    if seed is not None:
        env.seed(seed)
        
    return env


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


def evaluate(agent, env, max_steps=500, render=False):
    obs, i = env.reset()
    agent.model.reset()
    reward = 0
    steps = 0
    done = False
    if render:
        imgs = []
    while not done and steps < max_steps:
        if render:
            img = env.render() #mode='img'
            imgs.append(img)
        action = agent.act(obs)
        obs, r, done, trunc,  _ = env.step(action)
        reward += r
        steps += 1
        
    if render:
        return reward, imgs
    return reward


def generate_gif(gif_name='Robot.gif', a=None, env=None, solution_name=None):
    if solution_name is not None:
        a = load_solution(name=solution_name)
        cfg = a.config
        env = make_env(cfg["env_name"],
                       robot=cfg["robot"], render_mode="rgb_array")
    elif ((env is None) or (a is None)):
        raise ValueError("If there is no solution name, there must be both"
                         "an a and env.")
    
    env.metadata['render_fps'] = 50
    env.metadata.update({'render_modes': ["rgb_array"]})

    a.fitness, imgs = evaluate(a, env, max_steps=a.config["max_steps"],
                               render=True)
    env.close()

    print(a)

    imageio.mimsave(gif_name, imgs, duration=(1/50.0))


def ES(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"],
                  n=config["n"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    
    # Update weights
    mu = cfg["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i)
                          for i in range(1, mu + 1)])
    w /= np.sum(w)
    
    env = make_env(cfg["env_name"], robot=cfg["robot"])

    # Center of the distribution
    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = []
        for i in range(cfg["lambda"]):
            genes = theta + np.random.randn(len(theta)) * cfg["sigma"]
            ind = Agent(Network, cfg, genes=genes)
            population.append(ind)

        pop_fitness = [evaluate(a, env, max_steps=cfg["max_steps"]) 
                       for a in population]
        
        for i in range(len(population)):
            population[i].fitness = pop_fitness[i]

        # sort by fitness
        inv_fitnesses = [- f for f in pop_fitness]
        # indices from highest fitness to lowest
        idx = np.argsort(inv_fitnesses)
        
        step = np.zeros(d)
        for i in range(mu):
            # update step
            step = step + w[i] * (population[idx[i]].genes - theta)
        # update theta
        theta = theta + step * cfg["lr"]

        if pop_fitness[idx[0]] > elite.fitness:
            elite.genes = population[idx[0]].genes
            elite.fitness = pop_fitness[idx[0]]

        fits.append(elite.fitness)
        total_evals.append(len(population) * (gen+1))

        bar.set_description(f"Best: {elite.fitness}")
        
    env.close()
    
    if config['plot']:
        plt.figure()
        plt.plot(total_evals, fits)
        plt.xlabel("Evaluations")
        plt.ylabel("Fitness")
        plt.savefig(cfg["plot_name"])

    return elite


def CMAES(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"],
                  n=config["n"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    
    env = make_env(cfg["env_name"], robot=cfg["robot"])

    # Center of the distribution
    elite = Agent(Network, cfg)

    optimizer = CMA(mean=np.zeros(len(elite.genes)), sigma=cfg["sigma"],
                    population_size=cfg["lambda"])

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = []
        bar2 = tqdm(range(optimizer.population_size))
        bar2.set_description(f'gen #{gen+1}')

        for _ in bar2:
            genes = optimizer.ask()
            
            ind = Agent(Network, cfg, genes=genes)
            ind.fitness = evaluate(ind, env, max_steps=cfg["max_steps"])
            population.append((genes, -ind.fitness))

        optimizer.tell(population)

        elite.genes = optimizer.mean
        elite.fitness = evaluate(elite, env, max_steps=cfg["max_steps"])

        fits.append(elite.fitness)
        total_evals.append((len(population)+1) * (gen+1))

        bar.set_description(f"Best: {elite.fitness}")
        
    env.close()

    if config['plot']:
        plt.figure()
        plt.plot(total_evals, fits)
        plt.xlabel("Evaluations")
        plt.ylabel("Fitness")
        plt.savefig(cfg["plot_name"])

    return elite
