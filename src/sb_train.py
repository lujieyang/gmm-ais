import argparse
import gym
import gym_corridor

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
parser.add_argument("--A2C", action="store_true")
args = parser.parse_args()

# def env_maker(rank):
#     env = gym.make("CorridorNavigation-v0")
#     env.seed(rank)
#     env = Monitor(env)
#     return env

# env = gym.make("CorridorNavigation-v0")
# env = SubprocVecEnv([lambda: env_maker(i) for i in range(16)])
env = DummyVecEnv([lambda: Monitor(gym.make("CorridorNavigation-v0"))])
env = VecNormalize(env, norm_obs=False)

model_dir = "model/"

if args.A2C:
    model = A2C("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=int(5e4))
    model.save(model_dir + "sample_b_A2C_{}".format(args.seed))
else:
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, n_steps=8192)
    model.learn(total_timesteps=int(2.5e5))
    model.save(model_dir + "sample_b_PPO_{}".format(args.seed))

