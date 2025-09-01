import gymnasium as gym
import numpy as np
import testing
import utility
import player_testing
import curses

class FarkleController:

    dice_str = utility.get_dice_strings()
    lock_str = utility.get_lock_strings()


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
        self.log(f"Acknowledging farkle.")
        return self._env.acknowledge_farkle()

    def _bank_step(self):
        self.log(f"Acknowledging bank.")
        return self._env.acknowledge_bank()

    def play_turn(self, player, observation, info, reward, terminated, truncated):
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
        assert not terminated and not truncated

        if info["farkle"]:
            assert reward == -1
            self.log(f"Player {observation["turn"]} farkled off the bat! Sending reward to player.")
            # the player farkled off the bat
            player.update(observation, reward)
            # we prompt the environment to move to a new round for the next player's turn
            return self._farkle_step()

        action = {"bank": False}
        while not info["farkle"] and not action["bank"] and info["winner"] == -1 and not terminated and not truncated:
            assert reward == 0
            self.log(f"Prompting player {observation["turn"]} to play!")
            lock, bank = player.play(observation) # prompt current player to play
            action = {"lock": lock, "bank": bank}
            self.print_action(observation, action)
            observation, reward, terminated, truncated, info = self._env.step(action)
            self.log(f"Sending reward of {reward} to player {observation["turn"]}.")
            player.update(observation, reward)

        assert not truncated

        if terminated:
            assert info["winner"] != -1
            self.log(f"Player {observation["turn"]} won! They got {observation["points_this_turn"]} points this turn, bringing them to a total of {observation["player_points"][observation["turn"]]} points.")
            return observation, reward, terminated, truncated, info

        if info["farkle"]:
            self.log(f"Player {observation["turn"]} farkled! They would have got {observation["points_this_turn"]} points. They remain at {observation["player_points"][observation["turn"]]} points.")
            return self._farkle_step()
        elif action["bank"]:
            assert "lock" in action
            self.log(f"Player {observation["turn"]} banked! They got {observation["points_this_turn"]} points this turn, bringing them to a total of {observation["player_points"][observation["turn"]]} points.")
            return self._bank_step()
        else:
            raise Exception()

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
        reward = -1 if info["farkle"] else 0
        turns = 0 # TODO: only applicable to single player
        total_reward = 0

        while info["winner"] == -1 and not truncated and not terminated: # while game is not over TODO: consider truncated or terminated?
            current_player = observation["turn"]
            self.log(f"Start of player {current_player}'s turn.")
            observation, reward, terminated, truncated, info = self.play_turn(self.players[observation["turn"]], observation, info, reward, terminated, truncated)
            total_reward += reward
            turns += 1

        self.log(f"Winner is player {info['winner']}! It took a total of {turns} turns to win!")


if __name__ == "__main__":
    players = [player_testing.RandomPlayer()]
    env = testing.FarkleEnv()
    game = FarkleController(env, players)
    # observation = {"dice_values": [1,2,3,4,5,6], "dice_locked": [False, False, False, True, True, True], "turn": 1}
    # action = {"lock": [False, False, True, False, False, False], "bank": True}
    # game.print_dice(observation, action)
    # game.print_lock(observation, action)
    # game.print_bank(observation, action)
    # quit()
    for player in players:
        player.set_controller(game)
    game.play_game()

