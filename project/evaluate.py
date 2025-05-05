import evolve_tools as et
import os

# Common to all
seeds = [25, 30]
folder_name = os.path.join(os.path.dirname(__file__), 'final_solutions')

############
## Walker ##
############
file_name = os.path.join(folder_name, 'Walker-v0.json')
a = et.load_solution(name=file_name)
a.config["max_steps"] = 500
results = []

for seed in seeds:
    env = et.make_env("Walker-v0", robot=a.config["robot"],
                      seed=seed, render_mode='rgb_array')
    fitness = et.evaluate(a, env, max_steps=500)
    results.append((seed, fitness))
    et.generate_gif(gif_name=f"Walker_seed_{seed}.gif",a=a,
                    env=env, duration=(1/50.0), verbose=False)

print("\n\n\nFor the Walker-v0, our best robot had:")
for seed, fitness in results:
        print(f"\tfitness={fitness} in seed={seed}")

#############
## Thrower ##
#############
file_name = os.path.join(folder_name, 'Thrower-v0.json')
a = et.load_solution(name=file_name)
a.config["max_steps"] = 500
results = []

for seed in seeds:
    env = et.make_env("Thrower-v0", robot=a.config["robot"],
                      seed=seed, render_mode='rgb_array')
    fitness = et.evaluate(a, env, max_steps=500)
    results.append((seed, fitness))
    et.generate_gif(gif_name=f"Thrower_seed_{seed}.gif",a=a,
                    env=env, duration=(1/50.0), verbose=False)

print("\n\n\nFor the Thrower-v0, our best robot had:")
for seed, fitness in results:
        print(f"\tfitness={fitness} in seed={seed}")

#############
## Climber ##
#############
file_name = os.path.join(folder_name, 'Climber-v2.json')
a = et.load_solution(name=file_name)
a.config["max_steps"] = 500
results = []

for seed in seeds:
    env = et.make_env("Climber-v2", robot=a.config["robot"],
                      seed=seed, render_mode='rgb_array')
    fitness = et.evaluate(a, env, max_steps=500)
    results.append((seed, fitness))
    et.generate_gif(gif_name=f"Climber_seed_{seed}.gif",a=a,
                    env=env, duration=(1/50.0), verbose=False)

print("\n\n\nFor the Climber-v2, our best robot had:")
for seed, fitness in results:
        print(f"\tfitness={fitness} in seed={seed}")
