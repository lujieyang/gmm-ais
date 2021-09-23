import argparse
import gym
import gym_corridor

from stable_baselines3 import PPO, A2C, DQN

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
parser.add_argument("--A2C", action="store_true")
args = parser.parse_args()

env = gym.make("CorridorNavigation-v0")

model_dir = "model/"

if args.A2C:
    model = A2C("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=int(5e4))
    model.save(model_dir + "sample_b_A2C_{}".format(args.seed))
else:
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=int(5e4))
    model.save(model_dir + "sample_b_PPO_{}".format(args.seed))

