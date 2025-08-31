import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


# TODO: should we have a singler player environment wrapper? i dont know
class FarkleEnv(gym.Env):


    @staticmethod
    def _get_combinations():
        """
        Returns all valid scoring combinations in Farkle.
    
        Returns
        -------
        dict[int, list[dict[str, int]]]
            Dictionary mapping number of dice in a combination to 
            a list of scoring rule dictionaries. Each dictionary maps
            a *sorted* string representation of dice values (e.g., "111") to 
            the corresponding score.
        """

        singles = {"1": 100, "5": 50}
        doubles = {"11": 0, "22": 0, "33": 0, "44": 0, "55": 0, "66": 0} # a set, since pairs alone are worth nothing
        triples = {"111": 300, "222": 200, "333": 300, "444": 400, "555": 500, "666": 600}
        quadruples = {"1111": 1000, "2222": 1000, "3333": 1000, "4444": 1000, "5555": 1000, "6666": 1000} # a set, since quadruples are worth the same
        quintuples = {"11111": 2000, "22222": 2000, "33333": 2000, "44444": 2000, "55555": 2000, "66666": 2000}
        sextuples = {"111111": 3000, "222222": 3000, "333333": 3000, "444444": 3000, "555555": 3000, "666666": 3000}
        straight = {"123456": 1500}

        pair_keys = list(doubles.keys())
        three_pair_keys = [pair_keys[i] + pair_keys[j] + pair_keys[k] for i in range(4) for j in range(i+1, 5) for k in range(j+1, 6)]
        three_pair_value = 1500
        three_pair = dict.fromkeys(three_pair_keys, three_pair_value)

        triple_keys = list(triples.keys())
        two_triple_keys = [triple_keys[i] + triple_keys[j] for i in range(5) for j in range (i+1, 6)]
        two_triple_value = 2500
        two_triple = dict.fromkeys(two_triple_keys, two_triple_value)

        quadruple_keys = list(quadruples.keys())
        quadruple_and_pair_keys = [pair_keys[i] + quadruple_keys[j] for i in range(5) for j in range(i+1,6)]
        quadruple_and_pair_keys += [quadruple_keys[i] + pair_keys[j] for i in range(5) for j in range(i+1, 6)]
        quadruple_and_pair_value = 1500
        quadruple_and_pair = dict.fromkeys(quadruple_and_pair_keys, quadruple_and_pair_value)

        all_combinations = {1:[singles], 2:[], 3:[triples], 4:[quadruples], 5:[quintuples], 6:[sextuples, two_triple, quadruple_and_pair, straight, three_pair]} # exclude doubles because we never want to take doubles, and they don't count unless in combination with others

        return all_combinations
    
    combinations = _get_combinations()

    def __init__(self, players = 1, random_seed = None, max_points = 10000):
        # TODO: somehow establish a turn order, where we can report back to the agent which place in the turn order they get to play
            # furthermore, make this extensible to setting where we have multiple agents, not just one
            # need some kind of environment controller class to do this. Gymnasium not built for multiagent
        self.log("initializing FarkleEnv...")
        # number of players in the game
        self.players = players
        # number of dice 
        self.dice = 6
        # number of points to win the game
        self.max_points = max_points
        # set combinations to reduce runtime getting it in the future
        #self.combinations = self._get_combinations()

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

    def log(self, string):
        print(f"GAME: {string}")

    def _get_obs(self):
        """
        Get the current observation of the game state.
    
        Returns
        -------
        dict
            Dictionary with keys:
            - "dice_values": array of current die values
            - "dice_locked": array indicating locked dice
                0 if the corresponding dice is unlocked, 1 otherwise
            - "player_points": array of each player's total points
            - "turn": index of the current player's turn
            - "points_this_turn": points accumulated by the active player this turn
        """
        return {"dice_values": self._dice_values, "dice_locked": self._dice_locked, "player_points": self._player_points, "turn": self._turn, "points_this_turn": self._points_this_turn}

    def _get_info(self):
        """
        Returns additional info about the current state.
    
        Returns
        -------
        dict
            Contains:
            - "farkle": bool, whether current dice state is a Farkle
            - "winner": int, index of winning player if any, else -1
        """
        return {
            "farkle": self.check_farkle(self._dice_values, self._dice_locked), # TODO: this does not currently correctly return if the player farkled. no point, since the next player doesn't need this info
            "winner": self._check_win(),
        }

    def reset(self, seed = None, options = None):
        self.log("resetting FarkleEnv...")
        super().reset(seed=seed)

        # reset private representation of the game
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 
        self._player_points = np.array([0 for _ in range(self.players)], dtype=int)
        self._points_this_turn = 0
        self._turn = 0
        self._dice_values = self.observation_space["dice_values"].sample() # just sample to simulate the first dice roll of a game


        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # this is called to partially reset the environment state when a player ends their turn
    def _new_round(self):
        # partially reset private representation of the game
        # if next player farkles off the bat, move to next player's turn
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 
        self._points_this_turn = 0
        self._turn = (self._turn + 1) % self.players
        self._dice_values = self.observation_space["dice_values"].sample() # just sample to simulate the first dice roll of a game
        self.log(f"New round! Player {self._turn}, you're up!")

    def _roll_unlocked_dice(self):
        """
        Reroll all dice that are not locked, updating their values in place.
        """
        self.log("Rolling...")
        new_values = self.observation_space["dice_values"].sample()
        self._dice_values[self._dice_locked == 0] = new_values[self._dice_locked == 0]   # replace old dice values with new dice values in all indices where the dice are unlocked

    def _check_hot_dice(self, dice_locked):
        """
        Checks if all dice are locked ("hot dice").
    
        Parameters
        ----------
        dice_locked : array-like
            Binary array of locked dice. 0 in indices where the corresponding dice is unlocked, 1 otherwise
    
        Returns
        -------
        bool
            True if all dice are locked, False otherwise.
        """
        return np.all(dice_locked == 1)

    def _hot_dice(self):
        # partially reset private representation of dice
        # do not change turn or reset points_this_turn
        self.log("Hot dice!")
        self._dice_values = self.observation_space["dice_values"].sample()
        self._dice_locked = np.array([0 for _ in range(self.dice)], dtype=int) 

    def check_lock_legal(self, action):
        lock_action = action["lock"]
        assert len(lock_action) == self.dice
        for lock, already_locked in zip(lock_action, self._dice_locked):
            assert not (already_locked == 1 and lock == 1)

        if not all(x == 0 for x in lock_action):    # if the player is locking anything, make sure it's valid
            assert self.verify_combo(self._dice_values, lock_action)
            assert self.calculate_points(self._dice_values, lock_action) != 0

        return True

    def check_bank_legal(self, action):
        if action["bank"]:
            assert self._points_this_turn + self._player_points[self._turn] + self.calculate_points(self._dice_values, action["lock"]) >= 500

        return True

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
                if value == x and new_locked[i]:
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
            if points >= self.max_points:
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
            0 in indices where the corresponding dice is unlocked, 1 otherwise

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
        locked = [str(x) for x in locked]
        string = "".join(locked)
        max_points = 0
        for i in range(num_locked, 0, -1):
            for dict in FarkleEnv.combinations[i]:
                for key in dict.keys():
                    if key in string:
                        current_points = dict[key]
                        # TODO: is this supposed to be lock_action or dice_locked????
                        current_points += self.calculate_points(dice_values, self._helper_flip_lock(key, dice_values, lock_action)) #TODO: does this work?
                        max_points = max(current_points, max_points)

        return max_points

    def verify_combo(self, dice_values, lock_action):
        """
        verifies that the combination of dice a player has locked is valid.
        i.e., all dice locked correspond to one or multiple combinations

        Parameters
        ---------
        dice_values: array-like 
            the value of each die
        lock_action: array-like
            indicates which dice the player locked this turn
            0 in indices where the corresponding dice is unlocked, 1 otherwise

        Returns
        -------
        valid: bool
            True if the dice locked correspond to some valid combination, False otherwise
        """
        locked = []
        num_locked = 0
        for lock, die in zip(lock_action, dice_values):
            if lock:
                locked.append(die)
                num_locked += 1

        locked.sort()
        locked = [str(x) for x in locked]
        string = "".join(locked)
        for i in range(num_locked, 0, -1):
            for dict in FarkleEnv.combinations[i]:
                for key in dict.keys():
                    if key in string:
                        new_lock_action = self._helper_flip_lock(key, dice_values, lock_action)
                        if all(x == 0 for x in new_lock_action):
                            return True
                        if self.verify_combo(dice_values, new_lock_action): # if the currently found combination does not account for all the locked die, maybe this combination and some other with the remaining dice will
                            return True

        return False

    def check_farkle(self, dice_values, dice_locked):
        """
        checks if a player has farkled

        Parameters
        ---------
        dice_values: array-like 
            the value of each die
        dice_locked: array-like
            indicates if a die is locked or not
            0 in indices where the corresponding dice is unlocked, 1 otherwise

        Returns
        -------
        bool
            True is the player farkled
        """
        # do not check if a player has won.
        if np.any(self._player_points > self.max_points):
            return False
        # return True if player farkled, return False otherwise
        # TODO: use self._dice_values or allow parameter to be passed in?
        # TODO: use calculate_points for this?
        unlocked = []
        num_unlocked = 0
        for lock, die in zip(dice_locked, dice_values):
            if not lock:
                unlocked.append(die)
                num_unlocked += 1

        unlocked.sort()
        unlocked = [str(x) for x in unlocked]
        string = "".join(unlocked)
        for i in range(1, num_unlocked+1):
            for dict in FarkleEnv.combinations[i]:
                for key in dict.keys():
                    if key in string:
                        return False # the player did not farkle, there is at least one redeemable combination
        self.log(f"check_farkle found that Player {self._turn} farkled!")
        return True

    def check_legal(self, action):
        """
        checks if a player's action is legal
        """
        try:
            self.check_lock_legal(action)
        except AssertionError:
            self.log(f"Player {self._turn} attempted to lock illegal dice") 
            return False

        if action["bank"] and (self._player_points[self._turn] + self._points_this_turn + self.calculate_points(self._dice_values, action["lock"]) < 500):
            self.log(f"Player {self._turn} attempted to bank illegally")
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
        self.log("Entering step function")
        farkle = False
        terminated = False
        truncated = False
        # assert action["lock"] is not None and action["bank"] is not None        # player should never be prompted to play if they farkled
        if action["lock"] is None and action["bank"] is None:
            self.log("Game recognized that player farkled.")
            assert self.check_farkle(self._dice_values, self._dice_locked)
            # this is special to indicate that the player that has farkled has acknowledged this, and has been updated
            # now we reset to a new round
            self._new_round()
            reward = -1
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        assert self.check_legal(action)
        assert not self.check_farkle(self._dice_values, self._dice_locked)

        self.log("Updating locks")
        self._update_locks(action["lock"])

        points = self.calculate_points(self._dice_values, action["lock"]) # calculate the number of points scored by this action by using which dice were locked (THIS ACTION) by the player
        self.log(f"Player's action received {points}.")
        self._points_this_turn += points
        self.log(f"Player now has {self._points_this_turn} this turn.")

        if self._points_this_turn + self._player_points[self._turn] >= self.max_points:
            self.log(f"Player {self._turn} has over {self.max_points}! They win!")
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
                # TODO: do not enw round if player farkled, instead notify controller of farkle
                    # NO, DO BOTH
                self._new_round()
                farkle = True

        if action["bank"]:
            self._player_points[self._turn] += self._points_this_turn
            self._new_round() # only do new round if the player banked

        reward = 0 if (terminated or truncated or (not action["bank"] and not farkle)) else -1
        observation = self._get_obs()
        info = self._get_info() # TODO: add to info if turn ended?
        # TODO: add points this turn to info or observation?
        #assert not info["farkle"]
        return observation, reward, terminated, truncated, info
            
# register environment
register(
    id="gymnasium_env/FarkleEnv-v0",
    entry_point=FarkleEnv,
    max_episode_steps=500, # TODO: check if problem
    )

