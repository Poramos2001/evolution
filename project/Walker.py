import numpy as np
import evolve_tools as et


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
        "generations": 10, # to change: increase!
        "lambda": 10, # Population size
        "mu": 5, # Parents pop size
        "sigma": 0.1, # mutation std
        "lr": 1, # Learning rate
        "max_steps": 100, # to change to 500
        "plot_name": "ES.png",
        "n":3 # The network h_size = n*n_in, None sets h_size to 32
    }

    for i in range(3):
        print(f"\n\nRun #{i+1} with h_size = 3*env.observation_space.shape[0]")
        config["plot_name"] = f"Very Big Run #{i+1}"
        a = et.ES(config)
        print(a)

