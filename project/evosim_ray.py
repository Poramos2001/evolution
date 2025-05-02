import ray
import numpy as np
import time


class EvoGymEnv:
    def __init__(self, env_name, robot):
        import gymnasium as gym
        import evogym.envs
        self.env = gym.make(env_name, body=robot)
        self.env_name = env_name
        self.robot = robot
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.env_name, self.robot)
        return deserializer, serialized_data

    def reset(self):
        """
        Reset the environment and return the initial observation.
        """
        return self.env.reset()

    def step(self, action):
        """
        Take a step in the environment with the given action.
        """
        obs, reward, done, trunc,  info = self.env.step(action)
        return obs, reward, done, trunc, info


@ray.remote
def evaluate_env(env, horizon=1000):
    """
    Evaluate the environment for a given number of steps.
    """
    obs = env.reset()
    done = False
    value = 0
    for _ in range(horizon):
        action = env.action_space.sample()
        obs, reward, done, trunc,  info = env.step(action)
        value += reward
    return value


walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])
env = EvoGymEnv("Walker-v0", walker)

nb_evals = 16
#  Check time series
t_0 = time.time()
for _ in range(nb_evals):
    task = evaluate_env.remote(env, horizon=1000)
    result = ray.get(task)
t_1 = time.time()
print(f"Time taken for {nb_evals} evaluations in series:"
      f"{t_1 - t_0:.2f} seconds")

# Check time parallel
t_0 = time.time()
tasks = [evaluate_env.remote(env, horizon=1000) for _ in range(nb_evals)]
results = ray.get(tasks)
t_1 = time.time()

print(f"Time taken for {nb_evals} evaluations in parallel: "
      f"{t_1 - t_0:.2f} seconds")
