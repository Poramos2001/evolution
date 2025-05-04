import numpy as np
import evolve_tools as et


puncher = np.array([
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 2, 1],
    [0, 0, 4, 2, 5]
    ])

lever = np.array([
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [4, 0, 0, 0, 2, 4],
    [4, 0, 0, 0, 2, 4],
    [4, 0, 0, 0, 0, 5]
    ])

config = {
    "env_name": "Thrower-v0",
    "robot": lever,
    "generations": 70,
    "lambda": 80, # Population size
    "mu": 5, # Parents pop size
    "lr": 1, # Learning rate
    "max_steps": 100, # to change to 500
    "plot_name": "lever.png",
    "plot":True,
    "n": 3 # The network h_size = n*n_in, None sets h_size to 32
}

a = et.ES(config)
env = et.make_env(config["env_name"], robot=config["robot"],
                render_mode="rgb_array")
et.generate_gif(gif_name=f"lever.gif", a=a, env=env)