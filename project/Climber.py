import numpy as np
import evolve_tools as et
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter as CMAME
from ribs.schedulers import Scheduler

# Set up the archive
archive = GridArchive(
    dims=[100, 100],  # Dimensions of the archive grid
    ranges=[(-5, 5), (-5, 5)],  # Ranges for each dimension
    seed=42  # Random seed for reproducibility
)

# Configure the CMA-ES emitter
emitter = CMAME(
    archive,
    x0=np.zeros(2),  # Initial solution
    sigma0=0.5,  # Initial standard deviation
    batch_size=10  # Number of solutions to generate per iteration
)

# Set up the optimizer
optimizer = Scheduler(archive, [emitter])

# Run the MAP-Elites algorithm
for generation in range(100):  # Number of generations
    solutions = optimizer.ask()
    objective_values = np.array([objective_function(sol) for sol in solutions])
    optimizer.tell(objective_values)

# Retrieve the best solution
best_solution = archive.best_elite
print("Best solution:", best_solution.solution)
print("Best objective value:", best_solution.objective)



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