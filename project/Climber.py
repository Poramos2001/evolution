import numpy as np
import evolve_tools as et


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
    "generations": 180,
    "lambda": 50, # Population size
    "sigma": 4, # Initial std
    "mu": 5, # Parents pop size
    "lr": 1, # Learning rate
    "max_steps": 500, # to change to 500
    "plot_name": "climber.png",
    "plot":True,
    "n": 3 # The network h_size = n*n_in, None sets h_size to 32
}

a = et.ES(config)
env = et.make_env(config["env_name"], robot=config["robot"],
                render_mode="rgb_array")
et.generate_gif(gif_name=f"climber.gif", a=a, env=env)