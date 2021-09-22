from gym.envs.registration import register

register(
    id='CorridorNavigation-v0',
    entry_point='gym_corridor.envs:CorridorEnv',
)