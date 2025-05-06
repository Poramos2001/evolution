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
    
# ==== AJOUTS UTILITAIRES ======================================================
GRID_SHAPE = (5, 5)        #  <-- ici
N_MORPH     = GRID_SHAPE[0] * GRID_SHAPE[1]   # = 25
CELL_MAX = 5                   # valeurs 0-5 incluses
PENALTY = -1e6

def gene_to_body(g):           # g : (35,) float
    cells = np.clip(np.round(g), 0, CELL_MAX).astype(int)
    return cells.reshape(GRID_SHAPE)

def body_to_gene(body):        # body : (5,7) int
    return body.flatten().astype(np.float64)

def is_connected(body):
    """True si toutes les cellules ≠0 forment un pavé 4-connexe."""
    filled = (body != 0)
    idx = np.argwhere(filled)
    if len(idx) == 0:
        return False
    stack = [tuple(idx[0])]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<GRID_SHAPE[0] and 0<=nc<GRID_SHAPE[1] \
               and filled[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == filled.sum()

def actuator_idx(body):
    flat = body.flatten()
    return [i for i, v in enumerate(flat) if v in (3, 4)]
# ============================================================================

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
        self._morph = body_to_gene(config.get("robot",
                                        np.zeros(GRID_SHAPE, int)))  # NEW
        self.make_network()
        if genes is not None:
            self.genes = genes

    def make_network(self):
        n_in  = self.config["n_in"]
        h_sz  = self.config["h_size"]
        n_out = N_MORPH              # <-- toujours 35 sorties
        self.model = self.Net(n_in, h_sz, n_out).to(self.device).double()
        return self

    @property
    def genes(self):
        if self.model is None:
            return None
        # --- DETACH POUR ÉVITER L’ERREUR ---
        w_vec = (torch.nn.utils
                       .parameters_to_vector(self.model.parameters())
                       .detach()                 # <-- clé
                       .cpu()
                       .numpy())
        return np.concatenate([self._morph, w_vec])   # <-- morpho + weights

    @genes.setter
    def genes(self, vec):
        vec = np.asarray(vec, dtype=np.float64)
        self._morph, w_vec = vec[:N_MORPH], vec[N_MORPH:]
        if self.model is None:
            self.make_network()
        torch.nn.utils.vector_to_parameters(
            torch.tensor(w_vec, device=self.device), self.model.parameters())
        self.fitness = None

    # valeur par défaut : le corps initial passé dans config
    _morph: np.ndarray = body_to_gene(np.zeros(GRID_SHAPE, int))

    def act(self, obs):
        n_in = self.config["n_in"]          # 78
        # 1) Ajuster la longueur de l'observation
        if obs.shape[0] < n_in:             # pad à droite
            obs = np.pad(obs, (0, n_in - obs.shape[0]), 'constant')
        elif obs.shape[0] > n_in:           # tronque
            obs = obs[:n_in]

        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy().flatten()
        return actions

# ---------------------- Environment ----------------------


def make_env(env_name, seed=None, robot=None, **kwargs):
    import gymnasium as gym
    import evogym.envs

    if robot is None:                         # aucun corps fourni
        env = gym.make(env_name, **kwargs)
    else:
        arr = np.asarray(robot, dtype=int)    # assure ndarray
        connections = get_full_connectivity(arr)

        env = gym.make(
            env_name,
            body=arr,                         # <-- UN ARRAY, PAS DE DICT
            connections=connections,          # <-- UN ARRAY
            **kwargs
        )

    if seed is not None:
        env.reset(seed=seed)

    return env



# ---------------------- Evaluation ----------------------
def evaluate(agent, body, env, max_steps=500, render=False):
    idx_act = actuator_idx(body)
    if len(idx_act) == 0:          # sécurité supplémentaire
        return PENALTY

    obs, _ = env.reset()
    agent.model.reset()
    total = 0.0
    for _ in range(max_steps):
        if render:
            env.render()
        full_act = agent.act(obs)          # 35 scalaires
        act = full_act[idx_act]            # on ne garde que les n_actuateurs
        obs, r, done, trunc, _ = env.step(act)
        total += r
        if done:
            break
    return total

# ---------------------- Ray Remote Evaluation ----------------------
@ray.remote
def mp_eval_remote(genes, config):
    morph = gene_to_body(genes[:N_MORPH])
    # -- validité morpho --
    if not is_connected(morph) or len(actuator_idx(morph)) == 0:
        return PENALTY

    env = make_env(config["env_name"], robot=morph)
    agent = Agent(Network, config)
    agent.genes = genes                      # charge morpho+poids
    fit = evaluate(agent, morph, env, max_steps=config["max_steps"])
    env.close()
    return fit

# ---------------------- Evolution Strategy ----------------------
def get_cfg(env_name, robot=None):
    env = make_env(env_name, robot=robot if robot is not None else gene_to_body(np.zeros(N_MORPH)))
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": N_MORPH,          # <- 35 sorties
    }
    env.close()
    return cfg



def CMA_ES(config):
    ray.init(ignore_reinit_error=True)

    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg.update(config)

    config["n_in"] = cfg["n_in"]
    config["h_size"] = cfg["h_size"]
    config["n_out"] = cfg["n_out"]

    mu = cfg["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
    w /= np.sum(w)

    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)

    fits = []
    total_evals = []
    bar = tqdm(range(cfg["generations"]))

    for gen in bar:
        # Generate population
        genes_population = [theta + np.random.randn(d) * cfg["sigma"] for _ in range(cfg["lambda"])]

        # Ray async tasks
        futures = [mp_eval_remote.remote(genes, cfg) for genes in genes_population]
        pop_fitness = ray.get(futures)

        # Sort by fitness
        idx = np.argsort([-f for f in pop_fitness])

        step = np.zeros(d)
        for i in range(mu):
            step += w[i] * (genes_population[idx[i]] - theta)
        theta += step

        # Update elite
        if pop_fitness[idx[0]] > elite.fitness:
            elite.genes = genes_population[idx[0]]
            elite.fitness = pop_fitness[idx[0]]

        fits.append(elite.fitness)
        total_evals.append((gen + 1) * cfg["lambda"])

        bar.set_description(f"Best: {elite.fitness}")

    ray.shutdown()

    # Plot fitness curve
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Evolution Strategy Progress")
    plt.show()

    return elite

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
    
    thrower0 = np.array([
        [4, 0, 4, 1, 4],
        [3, 0, 4, 0, 3],
        [4, 0, 4, 0, 0],
        [4, 3, 4, 4, 2],
        [0, 4, 1, 0, 0]
    ])

    config = {
        "env_name": "Thrower-v0",             # <-- ici
        "robot":   thrower0,
        "generations": 300,
        "lambda": 40,
        "mu": 5, # Parents pop size
        "sigma": 0.25,
        "lr": 1, # Learning rate
        "max_steps": 300                      # <-- ici
    }

    a = CMA_ES(config)

    best_body = gene_to_body(a.genes[:N_MORPH])

    env = make_env(config["env_name"], robot=best_body)
    evaluate(a, best_body, env, render=False)
    env.close()

    np.save("Thrower.npy", a.genes)

    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
  


    save_solution(a, cfg)

    print(a.fitness)




    







    





