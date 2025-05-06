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
import ray
import random


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


class EvoGymEnv:
    def __init__(self, env_name, robot):
        import gymnasium as gym
        import evogym.envs
        self.env = gym.make(env_name, body=robot)
        self.env_name = env_name
        self.robot = robot
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.env_name, self.robot)
        return deserializer, serialized_data

    def reset(self):
        """
        Reset the environment and return the initial observation.
        """
        return self.env.reset()

    def step(self, action):
        """
        Take a step in the environment with the given action.
        """
        obs, reward, done, trunc,  info = self.env.step(action)
        return obs, reward, done, trunc, info
    
    def close(self):
        return self.env.close()


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
        env.reset(seed=seed)
        
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


def evaluate(agent, env, max_steps=500, render=False, seed=None):
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

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
        obs, r, done, _,  _ = env.step(action)
        reward += r
        steps += 1
        
    if render:
        return reward, imgs
    return reward


@ray.remote
def parallel_eval(agent, env, max_steps=500):
    obs, _ = env.reset()
    agent.model.reset()
    reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, r, done, _,  _ = env.step(action)
        reward += r
        steps += 1

    return reward


def generate_gif(gif_name='Robot.gif',
                 a=None,
                 env=None,
                 solution_name=None,
                 duration=(1/50.0),
                 verbose=True):

    if solution_name is not None:
        a = load_solution(name=solution_name)
        cfg = a.config
        env = make_env(cfg["env_name"],
                       robot=cfg["robot"], render_mode="rgb_array")
    elif ((env is None) or (a is None)):
        raise ValueError("If there is no solution name, there must be both"
                         "an a and env.")
    
    env.metadata['render_fps'] = 1/duration
    env.metadata.update({'render_modes': ["rgb_array"]})

    a.fitness, imgs = evaluate(a, env, max_steps=a.config["max_steps"],
                               render=True)
    env.close()

    if verbose:
        print(a)

    if imgs[0] is None:
        raise TypeError("The elements of imgs are None, check if the "
                        "environment 'env' provided was created with "
                        "render_mode='rgb_array'.")
    
    imageio.mimsave(gif_name, imgs, duration=duration)


def ES(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"],
                  n=config["n"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    
    # Update weights
    mu = cfg["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i)
                          for i in range(1, mu + 1)])
    w /= np.sum(w)
    
    env = EvoGymEnv(cfg["env_name"], cfg["robot"])

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
        sigma = (1-gen/cfg["generations"])*cfg["sigma"]
        for i in range(cfg["lambda"]):
            genes = theta + np.random.randn(len(theta)) * sigma
            ind = Agent(Network, cfg, genes=genes)
            population.append(ind)

        tasks = [parallel_eval.remote(a, env, max_steps=cfg["max_steps"]) 
                       for a in population]
        pop_fitness = ray.get(tasks)
        
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
    
    env = EvoGymEnv(cfg["env_name"], cfg["robot"])

    # Center of the distribution
    elite = Agent(Network, cfg)

    optimizer = CMA(mean=np.zeros(len(elite.genes)), sigma=cfg["sigma"],
                    population_size=cfg["lambda"])

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = []
        genetic_material = []
        bar2 = tqdm(range(optimizer.population_size))
        bar2.set_description(f'gen #{gen+1}')
            
        genomes = optimizer.ask()

        for genes in genomes:
            population.append(Agent(Network, cfg, genes=genes))
            genetic_material.append(genes)

        tasks = [parallel_eval.remote(a, env, max_steps=cfg["max_steps"]) 
                       for a in population]
        pop_fitness = ray.get(tasks)
        pop_fitness = [-x for x in pop_fitness]

        population = list(zip(genetic_material, pop_fitness))
        optimizer.tell(population)

        elite.genes = optimizer.mean
        tasks = parallel_eval.remote(elite, env, max_steps=cfg["max_steps"])
        elite.fitness = ray.get(tasks)
        
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    
    walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])

    config = {
        "env_name": "Walker-v0",
        "robot": walker,
        "generations": 10,
        "lambda": 10, # Population size
        "sigma": 1.3, # initial distribution variance
        "max_steps": 100, # to change to 500
        "plot_name": "CMAES.png",
        "plot":False,
        "n": None # The network h_size = n*n_in, None sets h_size to 32
    }

    for i in range(3):
        print(f"\n\nRun #{i+1} with n = {config['n']}")
        config["plot_name"] = f"Run #{i+1}"
        a = CMAES(config)
        print(a)



