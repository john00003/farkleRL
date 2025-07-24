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

    def _new_game(seed):
        return self._env.reset(seed)
        
    def play(seed):
        observation, info = self._new_game(seed)
        while info.winner == -1:
            

        pass

if __name__ == "__main__":
    print("what")

