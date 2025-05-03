import numpy as np
import evolve_tools as et


walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 1],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])

config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 10,
    "lambda": 10, # Population size
    "mu": 5, # Parents pop size
    "sigma": 0.1, # mutation std
    "lr": 1, # Learning rate
    "max_steps": 100, # to change to 500
    "plot_name": "ES.png",
    "plot":False,
    "n": 3 # The network h_size = n*n_in, None sets h_size to 32
}

a = et.ES(config)
env = et.make_env(config["env_name"], robot=config["robot"],
                  render_mode="rgb_array")
et.generate_gif(a=a, env=env)

# cfg = et.get_cfg(config["env_name"], robot=config["robot"],
#                   n=config["n"]) # Get network dims
# cfg = {**config, **cfg} # Merge configs
# et.save_solution(a, cfg)
