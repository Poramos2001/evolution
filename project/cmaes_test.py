import numpy as np
import evolve_tools as et


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
    a = et.CMAES(config)
    print(a)

