import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, Wrapper
import copy

class FarkleEnvSinglePlayerWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_players = len(env.players)
        # number of dice 
        self.dice = env.dice
        # number of points to win the game
        self.max_points = env.max_points

        # observation space of environment
            # value of each die
            # bool for each die - True if locked, False otherwise
            # points of each player
            # amount of points the player has scored already in their turn
        self.observation_space = gym.spaces.Dict(
            {
                "dice_values": gym.spaces.MultiDiscrete([6]*self.dice, seed=random_seed, start=[1]*self.dice),
                "dice_locked": gym.spaces.MultiBinary(self.dice),
                "player_points": gym.spaces.Box(0, max_points, shape=(1,), dtype=int),
                "points_this_turn": gym.spaces.Box(0, max_points, dtype=int)
            }
        )

    def observation(self, obs):
        """ this function wraps the original observation space to reduce player_points from a list to a single integer

        """
        return {"dice_values": obs["dice_values"], "dice_locked": obs["dice_locked"], "player_points": obs["player_points"][0], "points_this_turn": obs["points_this_turn"],


class FarkleReducedStateWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_players = len(env.players)
        # number of dice 
        self.dice = env.dice
        # number of points to win the game
        self.max_points = env.max_points

        # observation space of environment
            # value of each die
            # bool for each die - True if locked, False otherwise
            # points of each player
            # amount of points the player has scored already in their turn
        self.observation_space = gym.spaces.Dict(
            {
                "dice_values": gym.spaces.MultiDiscrete([6]*self.dice, seed=random_seed, start=[0]*self.dice),
                "player_points": gym.spaces.Box(0, max_points, shape=(num_players,), dtype=int),
                "points_this_turn": gym.spaces.Box(0, max_points, dtype=int)
            }
        )

        # action space of environment
            # bool - True if banking, False otherwise
            # bool for each die - True if locking, False otherwise
            # don't need a bool indicating if we roll! we roll all dice that are not locked if we don't bank
        self.action_space = gym.spaces.Dict(
            {
                "bank": gym.spaces.MultiBinary(1),
                "lock": gym.spaces.MultiBinary(self.dice)
            }
        )

    def _get_new_dice_representation(self, dice_values, dice_locked):
        dice_values = []
        for value, lock in zip(dice_values, dice_locked):
            if lock:
                dice_values.append(0)
            else:
                dice_values.append(value)

        dice_values.sort()
        return dice_values


    def observation(self, obs):
        dice_values = self._get_new_dice_representation(obs["dice_values"], obs["dice_locked"])

        return {"dice_values": dice_values, "player_points": obs["player_points"], "points_this_turn": obs["points_this_turn"]}

    def action(self, act):
        dice_copy = copy.deepcopy(self.env._dice_values)
        lock_copy = copy.deepcopy(self.env._dice_locked)
        new_rep = self._get_new_dice_representation(dice_copy, lock_copy)
        new_lock = [0] * len(dice_copy)

        dice_values_to_lock = []
        for value, lock in zip(dice_copy, lock_copy):
            if not lock:
                continue
            dice_values_to_lock.append(value)

        for i, die in enumerate(new_rep):
            if die in dice_values_to_lock:
                new_lock[i] = 1
                dice_values_to_lock.remove(die)
        
        return{"lock": new_lock, "bank": act["bank"]}
