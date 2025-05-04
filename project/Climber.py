import numpy as np
import evolve_tools as et


climber = np.array([
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 2, 1],
    [0, 0, 4, 2, 5]
    ])

config = {
    "env_name": "Climb-v2",
    "robot": climber,
    "generations": 100,
    "lambda": 10, # Population size
    "mu": 5, # Parents pop size
    "lr": 1, # Learning rate
    "max_steps": 100, # to change to 500
    "plot_name": "climber.png",
    "plot":True,
    "n": 3 # The network h_size = n*n_in, None sets h_size to 32
}

a = et.ES(config)
env = et.make_env(config["env_name"], robot=config["robot"],
                render_mode="rgb_array")
et.generate_gif(gif_name=f"climber.gif", a=a, env=env)