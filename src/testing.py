import numpy as np
import gymnasium as gym

class FarkleEnv(gym.Env):

    def __init__(self, players = 1, random_seed = None, max_points = 10000):
        # TODO: somehow establish a turn order, where we can report back to the agent which place in the turn order they get to play
            # furthermore, make this extensible to setting where we have multiple agents, not just one
        # number of players in the game
        self.players = players
        # number of dice 
        self.dice = 6
        # number of points to win the game
        self.max_points = max_points

        # TODO: use object oriented dice? or dice completely contained within environment?
            # can you do that with gym?


        # observation space of environment
            # value of each die
            # bool for each die - True if locked, False otherwise
            # points of each player
        self.observation_space = gym.spaces.Dict(
            {
                "dice_values": gym.spaces.Discrete(6, start=1, shape=(self.dice,), dtype=int, seed=random_seed),
                "dice_locked": gym.spaces.MultiBinary(self.dice),
                "player_points": gym.spaces.Box(0, max_points, shape=(self.players,), dtype=int),
                "points_this_turn": gym.spaces.Box(0, max_points, dtype=int)
            }
        )

        # private representation of the game
        self._dice_values = self.observation_space["dice_values"].sample()
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 
        self._player_points = np.array([0 for _ in range(self.players)], dtype=int)
        self._points_this_turn = 0
        self._turn = 0 

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

    def _get_obs(self):
        return {"dice_values": self._dice_values, "dice_locked": self._dice_locked, "player_points": self._player_points}

    def _get_info(self):
        return {
            "farkle": self.check_farkle(self._dice_values, self._dice_locked)
        }

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        # reset private representation of the game
        self._dice_values = self.observation_space["dice_values"].sample()
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 
        self._player_points = np.array([0 for _ in range(self.players)], dtype=int)
        self._points_this_turn = 0
        self._turn = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # this is called to partially reset the environment state when a player ends their turn
    def _new_round(self):
        # partially reset private representation of the game
        self._dice_values = self.observation_space["dice_values"].sample()
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 
        self._points_this_turn = 0
        self._turn = (self._turn + 1) % self.players
        # TODO: for multiple players, increment a turn counter

    def _hot_dice(self):
        # partially reset private representation of dice
        self._dice_values = self.observation_space["dice_values"].sample()
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 

    def check_lock_legal(self, lock_action):
        assert len(lock_action) == self.dice
        for lock, already_locked in zip(lock_action, self._dice_locked):
            assert (already_locked > lock) or (already_locked == 0)    # if die was already locked, assert player is not trying to lock it again.

    def _update_locks(self, lock_action):
        for i, lock in enumerate(lock_action):
            self._dice_locked[i] += lock

    def step(self, action):
        terminated = False
        points = self.calculate_points(self._dice_values, action["lock"]) # calculate the number of points scored by this action by using which dice were locked (THIS ACTION) by the player
        self._points_this_turn += points
        if self._points_this_turn + self._player_points[self._turn] >= self.max_points:
            terminated = True
            pass    # TODO: handle win
        if action["bank"]:
            # TODO: end turn, incur -1 reward for not winning
            self._player_points[self._turn] += self._points_this_turn
            self._new_round()

        # if player has not banked, do not end turn, do not give -1 reward


        # TODO: call new_round
        truncated = False
        reward = 0 if (terminated or not action["bank"]) else -1    # give -1 for every round until you win
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
            


