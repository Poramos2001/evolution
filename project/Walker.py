import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import evolve_tools as et
from cmaes import CMA

def CMAES(config):
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
            genes = optimizer.ask()
            print('asked #', i)
            
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
        "sigma": 1.3, # cma initialization
        "max_steps": 100,
    }

    a = CMAES(config)
    print("With CMA-ES")
    print(a)

    config = {
        "env_name": "Thrower-v0",
        "robot": walker,
        "generations": 10, # to change: increase!
        "lambda": 10, # Population size
        "mu": 5, # Parents pop size
        "sigma": 0.1, # mutation std
        "lr": 1, # Learning rate
        "max_steps": 100, # to change to 500
    }

    a = et.ES(config)
    print("With ES")
    print(a)

