import gymnasium as gym
import numpy as np
import testing
import player_testing
import curses

class FarkleController:

    @staticmethod
    def get_dice_strings():
        one = [" ----------- ",
            "|           |",
            "|           |",
            "|     0     |",
            "|           |",
            "|           |",
            " ----------- ",]
        two = [" ----------- ",
            "| 0         |",
            "|           |",
            "|           |",
            "|           |",
            "|         0 |",
            " ----------- ",]
        three = [" ----------- ",
            "| 0         |",
            "|           |",
            "|     0     |",
            "|           |",
            "|         0 |",
            " ----------- ",]
        four = [" ----------- ",
            "| 0       0 |",
            "|           |",
            "|           |",
            "|           |",
            "| 0       0 |",
            " ----------- ",]
        five = [" ----------- ",
            "| 0       0 |",
            "|           |",
            "|     0     |",
            "|           |",
            "| 0       0 |",
            " ----------- ",]
        six = [" ----------- ",
            "| 0       0 |",
            "|           |",
            "| 0       0 |",
            "|           |",
            "| 0       0 |",
            " ----------- ",]
        dice_str = {1: one, 2: two, 3: three, 4: four, 5: five, 6: six}

        return dice_str

    @staticmethod
    def get_lock_strings():
        locked = "     [x]     " 
        newly_locked = "     [X]     " 
        unlocked = "     [ ]     "
        lock_str = {0: unlocked, 1: locked, 2: newly_locked}
        return lock_str

    dice_str = get_dice_strings()
    lock_str = get_lock_strings()


    def __init__(self, env, players, agent_player_num = 0):
        """
        initializes the FarkleController class

        Parameters
        ----------
        env: FarkleEnv
            the gymnasium environment that will be played
        players: array-like
            a list of player objects
        agent_player_num: int
            the index in the players of the agent that is training
        """
        assert agent_player_num < len(players)
        self._env = env
        self.players = players
        self.agent_player_num = agent_player_num # TODO: can we get rid of this?

    def log(self, string):
        print(f"CONTROLLER: {string}")

    def print_action(self, observation, action):
        self.print_dice(observation, action)
        self.print_lock(observation, action) #TODO: print_lock should have some indicator (such as ^) to indicate dice was locked this turn
        self.print_bank(observation, action)

    def print_dice(self, observation, action):
        dice = [FarkleController.dice_str[x] for x in observation["dice_values"]]
        
        for i in range(7):
            line = [die[i] for die in dice]
            print(f"{"  ".join(line)}")

    def print_lock(self, observation, action):
        locked = [FarkleController.lock_str[2] if action["lock"][i] else FarkleController.lock_str[1] if observation["dice_locked"][i] else FarkleController.lock_str[0] for i in range(len(observation["dice_locked"]))]
        print(f"{"  ".join(locked)}")

    def print_bank(self, observation, action):
        if action["bank"]:
            print(f"             PLAYER {observation["turn"]} BANKED!                 ")

    def _new_game(self, seed = None):
        """
        Start a new game in the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        observation : dict
            Initial observation from the environment.
        info : dict
            Additional info returned by the environment.
        """
        return self._env.reset(seed)

    def check_legal(self, action):
        return self._env.check_legal(action)

    def check_lock_legal(self, action):
        try:
            return self._env.check_lock_legal(action)
        except AssertionError:
            return False

    def check_bank_legal(self, action):
        try:
            return self._env.check_bank_legal(action)
        except AssertionError:
            return False

    def _farkle_step(self):
        action = {"lock": None, "bank": None}
        return self._env.step(action)

    def play_turn(self, player, observation, info):
        """
        Play a full turn for a given player.

        Parameters
        ----------
        player : Player
            Player object whose turn it is.
        observation : dict
            Current observation from the environment.
        info : dict
            Additional environment info.

        Returns
        -------
        observation : dict
            Final observation after the player's turn. A dict containing:
                - "dice_values": array of current die values
                - "dice_locked": array indicating locked dice
                    0 if the corresponding dice is unlocked, 1 otherwise
                - "player_points": array of each player's total points
                - "turn": index of the current player's turn
                - "points_this_turn": points accumulated by the active player this turn
        reward : float
            Reward obtained during the turn.
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode was truncated.
        info : dict
            Final info dictionary after the turn.
        """
        reward_this_turn = 0
        player_num = observation["turn"]
        truncated = False
        terminated = False

        # first check if we farkled off the bat
        if info["farkle"]:
            self.log(f"Player {player_num} farkled! Sending farkle action...")
            observation, reward, terminated, truncated, info = self._farkle_step() # call special function for when player farkled before they went
            return observation, reward, terminated, truncated, info

        while(observation["turn"] == player_num and not truncated and not terminated): # until we move to next player
            self.log(f"Prompting player {player_num} to play...")
            lock, bank = player.play(observation) # prompt current player to play
            self.log(f"Player {player_num} played lock: {lock}, bank: {bank}")
            action = {"lock": lock, "bank": bank}
            observation, reward, terminated, truncated, info = self._env.step(action)
            self.log(f"Player {player_num} received {reward} from that action")
            reward_this_turn += reward
            if info["farkle"] or bank:
                if info["farkle"]:
                    self.log(f"Player {player_num} is ending their turn because they farkled.")
                else:
                    self.log(f"Player {player_num} is ending their turn because they are banking")
                break

        # TODO: update player with reward after every action, or after all actions?
            # well, there will be only be one reward, the -1 that we get at the very end of the turn
        self.log(f"End of player {player_num}'s turn. They received {observation["points_this_turn"]} points this turn. Player now has {observation["player_points"][player_num]} points.")
        self.log(f"Rewarding player {player_num} with reward of {reward_this_turn}.")
        player.update(reward_this_turn)

        # return what is returned by step()
        return observation, reward, terminated, truncated, info
        
    def play_game(self, seed = None):
        """
        Play a complete game of Farkle until a winner is determined.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        winner : int
            Index of the winning player.
        """
        observation, info = self._new_game(seed)
        truncated = False
        terminated = False

        # TODO: important for controller not to determine who is next to play. let FarkleEnv and observation tell us who is next to play (in case of b2b farkles)
        while info["winner"] == -1 and not truncated and not terminated: # while game is not over TODO: consider truncated or terminated?
            current_player = observation["turn"]
            self.log(f"Start of player {current_player}'s turn.")
            observation, reward, terminated, truncated, info = self.play_turn(self.players[observation["turn"]], observation, info)

        self.log(f"Winner is player {info['winner']}!")


if __name__ == "__main__":
    players = [player_testing.RandomPlayer()]
    env = testing.FarkleEnv()
    game = FarkleController(env, players)
    observation = {"dice_values": [1,2,3,4,5,6], "dice_locked": [False, False, False, True, True, True], "turn": 1}
    action = {"lock": [False, False, True, False, False, False], "bank": True}
    game.print_dice(observation, action)
    game.print_lock(observation, action)
    game.print_bank(observation, action)
    quit()
    for player in players:
        player.set_controller(game)
    game.play_game()

