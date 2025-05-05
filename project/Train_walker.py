import numpy as np
import evolve_tools as et


walker = np.array([
    [3, 3, 3, 3, 3, 3],
    [4, 2, 2, 2, 2, 4],
    [3, 3, 0, 0, 3, 3],
    [3, 3, 0, 0, 3, 3],
    [3, 3, 0, 0, 3, 3]
    ])

config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 800,
    "lambda": 20, # Population size
    "mu": 5, # Parents pop size
    "lr": 1, # Learning rate
    "max_steps": 500, # to change to 500
    "plot_name": "Final_walker.png",
    "plot":True,
    "n": 3 # The network h_size = n*n_in, None sets h_size to 32
}

for i in range(3):
    config["plot_name"] = f"final_walker_{i+1}.png"

    print(f"\n\n\nfinal_walker_{i+1}")
    a = et.ES(config)
    env = et.make_env(config["env_name"], robot=config["robot"],
                    render_mode="rgb_array")
    et.generate_gif(gif_name=f"final_walker_{i+1}.gif", a=a, env=env)

    cfg = et.get_cfg(config["env_name"], robot=config["robot"],
                    n=config["n"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    et.save_solution(a, cfg, name=f"final_walker_{i+1}.json")
