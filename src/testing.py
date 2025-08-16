import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


# TODO: should we have a singler player environment wrapper? i dont know
class FarkleEnv(gym.Env):


    @staticmethod
    def _get_combinations():

        # singles
        all_combinations = {(0, 0, 0, 0, 0, 1) : 100, (0, 0, 0, 0, 0, 5) : 50}

        for i in range(1, 7):
            # 3, 4, 5, 6 of same kind
            all_combinations[(0, 0, 0, i, i, i)] = i * 100 if i > 1 else 300
            all_combinations[(0, 0, i, i, i, i)] = 1000
            all_combinations[(0, i, i, i, i, i)] = 2000
            all_combinations[(i, i, i, i, i, i)] = 3000
            
            for j in range(i + 1, 7):
                for k in range(j + 1, 7):
                    # three pairs
                    all_combinations[(i, i, j, j, k, k)] = 1500


        for i in range(1, 7):
            for j in range(1, 7):

                # four of a kind and pair
                temp = tuple(sorted([i, i, i, i, j, j]))

                if temp not in all_combinations:
                    all_combinations[temp] = 1500

                if j < i or i == j:
                    continue
                #two triplets
                temp = (i, i, i, j, j, j)

                if temp not in all_combinations:
                    all_combinations[temp] = 2500

        # straight
        all_combinations[(1, 2, 3, 4, 5, 6)] = 1500

        return all_combinations
    
    def __init__(self, players = 1, random_seed = None, max_points = 10000):
        # TODO: somehow establish a turn order, where we can report back to the agent which place in the turn order they get to play
            # furthermore, make this extensible to setting where we have multiple agents, not just one
            # need some kind of environment controller class to do this. Gymnasium not built for multiagent
        # number of players in the game
        self.players = players
        # number of dice 
        self.dice = 6
        # number of points to win the game
        self.max_points = max_points
        # set combinations to reduce runtime getting it in the future
        self.combinations = self._get_combinations()

        # observation space of environment
            # value of each die
            # bool for each die - True if locked, False otherwise
            # points of each player
            # amount of points the player has scored already in their turn
        self.observation_space = gym.spaces.Dict(
            {
                "dice_values": gym.spaces.MultiDiscrete([6]*self.dice, seed=random_seed, start=[1]*self.dice),
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
        return {"dice_values": self._dice_values, "dice_locked": self._dice_locked, "player_points": self._player_points, "turn": self._turn, "points_this_turn": self._points_this_turn}

    def _get_info(self):
        return {
            "farkle": self.check_farkle(self._dice_values, self._dice_locked),
            "winner": self._check_win(),
        }

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        # reset private representation of the game
        self._dice_values = self.observation_space["dice_values"].sample() # just sample to simulate the first dice roll of a game
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
        # if next player farkles off the bat, move to next player's turn
        while True:
            self._dice_values = self.observation_space["dice_values"].sample()
            self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 
            self._points_this_turn = 0
            self._turn = (self._turn + 1) % self.players
            if not self.check_farkle(self._dice_values, self._dice_locked):
                break

    def _roll_unlocked_dice(self):
        """
        rolls the dice that are unlocked
        """
        new_values = self.observation_space["dice_values"].sample()
        self._dice_values[self._dice_locked == 0] = new_values[self.dice_locked == 0]   # replace old dice values with new dice values in all indices where the dice are unlocked

    def _check_hot_dice(self, dice_locked):
        return np.all(dice_locked == 1)

    def _hot_dice(self):
        # partially reset private representation of dice
        # do not change turn or reset points_this_turn
        self._dice_values = self.observation_space["dice_values"].sample()
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 

    def check_lock_legal(self, lock_action):
        assert len(lock_action) == self.dice
        for lock, already_locked in zip(lock_action, self._dice_locked):
            assert (already_locked > lock) or (already_locked == 0)    # if die was already locked, assert player is not trying to lock it again.

    def _update_locks(self, lock_action):
        for i, lock in enumerate(lock_action):
            self._dice_locked[i] += lock

    def _helper_flip_lock(self, string, dice_values, dice_locked):
        """
        returns a new array of which dice are locked after the player has attempted to lock a combination of dice

        Parameters
        ----------
        string: string
            a string indicating the values of the dice the player is trying to lock
        dice_values: array-like
            an array of integers indicating the value of each die in each position
        dice_locked: array-like
            0 if the die is unlocked, 1 otherwise

        Returns
        -------
        new_locked: array-like
            0 if the die was previously locked, but we are unlocking it by redeeming some combination of points, 1 otherwise
        """
        new_locked = [x for x in dice_locked]
        for char in string:
            x = int(char)
            for i, value in enumerate(dice_values): # we find a dice of matching value and undo the lock
                if value == x and dice_locked[i]:
                    new_locked[i] = 0
                    break
        return new_locked

    # checks if any player has win, returning the player number if so, -1 otherwise
    def _check_win(self):
        """
        checks if a player has won

        Returns
        -------
        int
            the player number if somebody has won, -1 otherwise
        """
        for i, points in enumerate(self._player_points):
            if points > self.max_points:
                return i
        return -1


    def calculate_points(self, dice_values, lock_action):
        """
        checks how many points a player obtained with the dice they locked

        Parameters
        ---------
        dice_values: array-like 
            the value of each die
        lock_action: array-like
            indicates which dice the player locked this turn

        Returns
        -------
        max_points: integer
            the amount of points scored by the player's lock actions
        """
        # TODO: check_farkle and check_legal lock can also be implemented here (for game functionality)
            # just check if points == 0 and if so, the player farkled?
        # TODO: add to check legal lock that valid combinations are locked!


        locked = []
        num_locked = 0
        for lock, die in zip(lock_action, dice_values):
            if lock:
                locked.append(die)
                num_locked += 1

        locked.sort()
        string = "".join(locked)
        max_points = 0
        for i in range(num_locked, 0, -1):
            for dict in FarkleEnv.combinations[i]:
                for key in dict.keys():
                    if key in string:
                        current_points = dict[key]
                        current_points += calculate_points(dice_values, self._helper_flip_lock(key, dice_values, lock_action)) #TODO: does this work?
                        max_points = max(current_points, max_points)

        return max_points

    # Proposol, when die is locked we just set the value to 0
    def check_farkle(self, dice_values):
        """
        checks if a player has farkled

        Parameters
        ---------
        dice_values: a tupple
            the value of each die
        Returns
        -------
        bool
            True is the player farkled
        """

        # If the dice_values are not in the combination ur cooked
        return dice_values not in self.combinations

    def check_legal(self, action):
        """
        checks if a player's action is legal
        """
        try:
            self.check_lock_legal(action.lock)
        except AssertionError:
            return False

        if action.bank and (self._player_points[self._turn] + self._points_this_turn < 500):
            return False

        return True

    def step(self, action):
        """
        step is a function to implement Gymnasium's environment API

        Parameters
        ---------
        action: dict
            contains "lock": an array-like with a 1 in each index where the player would like to lock the dice,
                     "bank": a boolean indicating if the player is banking after this turn

        Returns
        -------
        observation: dict
            contains "dice_values": an array-like with integer elements indicating the new value of each die
                     "dice_locked": an array-like with a 1 in each index where a die is locked
                     "player_points": an array-like with the points of each player (not including any potential points by the current player this turn)
                     "turn": an integer indicating who's turn it is
        """
        # TODO: okay absolutely under no circumstances do you prompt the player for action if they farkled
            # can we add info for controller to read - perhaps if current player farkled due to this turn, we need to indicate to controller that it is next player's turn
            # but what to do if upon rolling initial dice for next player, the next player farkles as well?
                # well, thankfully we don't need to give any reward for that. other player that farkled did not necessarily "take an action"
                    # just do a while loop, new round until no more farkling!
        # TODO: we need to reroll remaining dice in this function somewhere
        assert action["lock"] is not None and action["bank"] is not None        # player should never be prompted to play if they farkled
        assert self.check_legal(action)
        assert not self.check_farkle(self._dice_values, self._dice_locked)

        if not action["bank"]:
            # no need to update the locks if player is banking
            self._update_locks(action["lock"])

        farkle = False
        terminated = False
        truncated = False
        points = self.calculate_points(self._dice_values, action["lock"]) # calculate the number of points scored by this action by using which dice were locked (THIS ACTION) by the player
        self._points_this_turn += points

        if self._points_this_turn + self._player_points[self._turn] >= self.max_points:
            terminated = True
            self._player_points[self._turn] += self._points_this_turn
            reward = 0
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        else:
            terminated = False

        hot_dice = self._check_hot_dice(self._dice_locked)
        if not action["bank"]:
            if not hot_dice:
                self._roll_unlocked_dice() # we
            else:
                self._hot_dice()
            # in both of these cases, player may have farkled
            if self.check_farkle(self._dice_values, self._dice_locked):
                # player farkled, give them negative reward
                self._new_round()
                farkle = True

        # this call to check farkle cannot be legal. plus, the environment already checked if the player farkled before the agent made any actions
        # and we are passing only the dice that the player LOCKED THIS TURN to check farkle, when we should be passing (prev locked die OR die locked this turn)
        # if self.check_farkle(self._dice_values, action["lock"]):
        #     assert points == 0
        #     self._points_this_turn = 0
            # self._new_round()
        #elif action["bank"]:
        if action["bank"]:
            self._player_points[self._turn] += self._points_this_turn
            self._new_round() # only do new round if the player banked

        reward = 0 if (terminated or truncated or (not action["bank"] and not farkle)) else -1
        #reward = 0 if (terminated or (not action["bank"] and not farkle) or (action["bank"] and hot_dice)) else -1    # give -1 for every round until you win
        print(reward)   #confirm that reward is correct
        observation = self._get_obs()
        info = self._get_info() # TODO: add to info if turn ended?
        # TODO: add points this turn to info or observation?
        assert not info.farkle
        return observation, reward, terminated, truncated, info
            
# register environment
register(
    id="gymnasium_env/FarkleEnv-v0",
    entry_point=FarkleEnv,
    max_episode_steps=500, # TODO: check if problem
    )


### Add main function to test
if __name__ == "__main__":
    test = FarkleEnv()
