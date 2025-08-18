import gymnasium as gym
import numpy as np
import testing
import player_testing

class FarkleController:

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

        while(observation["turn"] == player_num and not truncated and not terminated): # until we move to next player
            print(f"Prompting player {player_num} to play...")
            lock, bank = player.play(observation) # prompt current player to play
            action = {"lock": lock, "bank": bank}
            observation, reward, terminated, truncated, info = self._env.step(action)
            print(f"Player {player_num} received {reward} from that action")
            reward_this_turn += reward

        # TODO: update player with reward after every action, or after all actions?
            # well, there will be only be one reward, the -1 that we get at the very end of the turn
        print(f"Rewarding player {player_num} with reward of {reward_this_turn}.")
        player.update(reward_this_turn)

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
            print(f"Start of player {current_player}'s turn.")
            observation, reward, terminated, truncated, info = self.play_turn(self.players[observation["turn"]], observation, info)

        print(f"Winner is player {info['winner']}!")


if __name__ == "__main__":
    players = [player_testing.RandomPlayer()]
    env = testing.FarkleEnv()
    game = FarkleController(env, players)
    for player in players:
        player.set_controller(game)
    game.play_game()

