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
import imageio


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
GRID_SHAPE = (5, 5)            # (rows, cols)  = 35 cellules
N_MORPH = GRID_SHAPE[0] * GRID_SHAPE[1]
CELL_MAX = 4                  # valeurs 0-5 incluses
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
    if robot is None:
        env = gym.make(env_name)
    else:
        connections = get_full_connectivity(robot)
        env = gym.make(env_name, body=robot, connections=connections, **kwargs)
    if seed is not None:
        env.seed(seed)
    return env



# ---------------------- Evaluation ----------------------
def evaluate(agent, body, env, max_steps=500, render=False):
    idx_act = actuator_idx(body)
    if len(idx_act) == 0:          # sécurité supplémentaire
        return PENALTY

    obs, _ = env.reset()
    agent.model.reset()
    total = 0.0
    if render:
        imgs = []
    for _ in range(max_steps):
        if render:
            img = env.render() #mode='img'
            imgs.append(img)
        full_act = agent.act(obs)          # 35 scalaires
        act = full_act[idx_act]            # on ne garde que les n_actuateurs
        obs, r, done, trunc, _ = env.step(act)
        total += r
        if done:
            break
    
    if render:
        return total, imgs
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
def get_cfg(env_name, robot=None, n=None):
    env = make_env(env_name, robot=robot if robot is not None else gene_to_body(np.zeros(N_MORPH)))
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": N_MORPH,          # <- 35 sorties
    }
    if n is not None:
        cfg["h_size"] = n*cfg["n_in"]
    env.close()
    return cfg



def CMA_ES(config):
    ray.init(ignore_reinit_error=True)

    cfg = get_cfg(config["env_name"], robot=config["robot"], n=config["n"])
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
    plt.figure()
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.savefig(cfg["plot_name"])

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
    [0, 4, 4, 2, 0],
    [0, 2, 4, 4, 0],
    [1, 4, 1, 2, 1],
    [3, 3, 1, 3, 3]
    ])

    config = {
        "env_name": "Climber-v2",
        "robot": climber,
        "generations": 100,
        "lambda": 30, # Population size
        "sigma": 3, # initial distribution variance
        "max_steps": 500, # to change to 500
        "plot_name": "climber.png",
        "plot":True,
        "n": 1 # The network h_size = n*n_in, None sets h_size to 32
    }

    lambdas = [30, 40, 50]
    sigmas =[0.7, 1.3, 2, 3]
    ns = [1, 2]

    for n in ns:
        config["n"] = n
        for sigma in sigmas:
            config["sigma"] = sigma
            for lam in lambdas:
                config["lambda"] = lam
                config["plot_name"] = f"(n, 10*sigma, lambda) = ({n}, {10*sigma}, {lam}).png"
                
                a = CMA_ES(config)

                cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
                cfg = {**config, **cfg} # Merge configs
                save_solution(a, cfg, name=f"({n},{sigma},{lam}).json")

                best_body = gene_to_body(a.genes[:N_MORPH])

                env = make_env(config["env_name"], robot=best_body, render_mode='rgb_array')
                env.metadata['render_fps'] = 50.0
                env.metadata.update({'render_modes': ["rgb_array"]})

                a.fitness, imgs = evaluate(a, best_body, env, render=True)
                env.close()
                imageio.mimsave(f"({n},{10*sigma},{lam}).gif", imgs, duration=(1/50.0))




    







    





