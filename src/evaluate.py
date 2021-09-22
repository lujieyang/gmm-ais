import os
import torch
import copy
import argparse

from PPO import ActorCritic
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from Experiments.GetTestParameters import *

def eval(seed, n_episodes=100, gamma=0.95):
    env_name = "CorridorNavigation"
    has_continuous_action_space = False
    action_std = 0.6
    random_seed = seed

    print("random seed: ", random_seed)

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder
    # state space dimension
    state_dim = 30
    # action space dimension
    action_dim = 3

    ncBelief = 10
    POMDP, P = GetTestParameters0(ncBelief=ncBelief)
    S = POMDP.S
    start = P["start"]

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    returns = []
    for n_eps in range(n_episodes):
        print(".", end=" ")

        b = copy.deepcopy(start)
        s = S.Crop(b.rand())
        state = b.to_array()
        reward_episode = []

        for t in range(30):

            # select action with policy
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = policy.act(state)
            action = action.item()
            s, b, _, reward, rb, bn, _ = POMDP.SimulationStep(b, s, action+1)
            state = b.to_array()

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
    return average_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
    args = parser.parse_args()
    # set device to cpu or cuda
    device = torch.device('cpu')

    if(torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    eval(args.seed)