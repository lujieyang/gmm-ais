import os
import torch
import argparse
import numpy as np

import gym
import gym_corridor

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import obs_as_tensor

def eval(args, seed, n_episodes=100, gamma=0.95):
    env = gym.make("CorridorNavigation-v0")
    model_dir = "model/"
    print("seed: ", seed)
    if args.A2C:
        model = A2C.load(model_dir+"sample_b_A2C_{}".format(seed), env=env)
    else:
        model = PPO.load(model_dir+"sample_b_PPO_{}".format(seed), env=env)

    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        reward_episode = []

        for t in range(30):

            # select action with policy
            actions, _ = model.predict(obs)
            obs, reward, _, _ = env.step(actions)

            reward_episode.append(reward)

        rets = []
        R = 0
        for i, r in enumerate(reward_episode[::-1]):
            R = r + gamma * R
            rets.insert(0, R)
        returns.append(rets[0])

    print("\n")
    average_return = np.mean(returns)
    print("Average reward: ", average_return)
    print("Reward std: ", np.std(returns))
    return returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
    parser.add_argument("--A2C", action="store_true")
    parser.add_argument("--group",action="store_true")
    args = parser.parse_args()
    # set device to cpu or cuda
    device = torch.device('cpu')

    if(torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    if args.group:
        returns = []
        seed = [2048, 67, 88, 72, 77, 512, 42, 10, 1024, 32] #[1, 2, 3, 4, 5, 6, 42, 10, 1024, 32]
        for s in seed:
            returns.append(eval(args, s))
        returns = np.array(returns)
        np.save("model/PPO_return", returns)
    else:
        eval(args, args.seed)

    