import numpy as np
from cmaes import CMA
import matplotlib.pyplot as plt
from tqdm import tqdm
import evolve_tools as et

def debug_CMAES(config):
    cfg = et.get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    
    env = et.make_env(cfg["env_name"], robot=cfg["robot"])

    # Center of the distribution
    elite = et.Agent(et.Network, cfg)

    optimizer = CMA(mean=np.zeros(len(elite.genes)), sigma=cfg["sigma"],
                    population_size=cfg["lambda"])

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = []
        print('gen #', gen)
        for i in range(optimizer.population_size):
            print('is going to ask')
            try:
                genes = optimizer.ask()
                print('asked #', i)
            except Exception as e:
                print(f"Error during optimizer.ask() at generation {gen}, individual {i}: {e}")
                return
            
            ind = et.Agent(et.Network, cfg, genes=genes)
            ind.fitness = et.evaluate(ind, env, max_steps=cfg["max_steps"])
            population.append((genes, -ind.fitness))

        optimizer.tell(population)

        elite.genes = optimizer.mean
        elite.fitness = et.evaluate(elite, env, max_steps=cfg["max_steps"])

        fits.append(elite.fitness)
        total_evals.append((len(population)+1) * (gen+1))

        bar.set_description(f"Best: {elite.fitness}")
        
    env.close()

    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.show()

    return elite

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
    "sigma": 1.3, # cma initialization
    "max_steps": 100,
}

a = debug_CMAES(config)
print(a)

