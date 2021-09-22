import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import copy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from Experiments.GetTestParameters import *

class CorridorEnv(gym.Env):
  def __init__(self):
    ncBelief = 10
    POMDP, P = GetTestParameters0(ncBelief=ncBelief)
    self.observation_dim = 1000 #ncBelief*3
    self.POMDP = POMDP
    self.P = P
    self.S = POMDP.S
    self.start = P["start"]
    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(low=-22, high=22,shape=(self.observation_dim,))
    self.step_num = 0
    self.step_ind = 30

  def step(self, action):
    done = False
    s, b, _, reward, rb, bn, _ = self.POMDP.SimulationStep(self.b, self.s, action+1)
    self.s = s
    self.b = b
    self.step_num += 1
    if self.step_num == self.step_ind:
      done = True
    return b.sample_array(), reward, done, {}

  def reset(self):
    self.b = copy.deepcopy(self.start)
    self.s = self.S.Crop(self.b.rand())
    self.step_num = 0
    return self.b.sample_array()

  def render(self):
    pass