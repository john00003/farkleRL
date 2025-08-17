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
        return self._env.reset(seed)

    def check_legal(self, action):
        return env.check_legal(action)

    def check_lock_legal(self, action):
        print(f"action in controller: {action}")
        print(env.check_lock_legal(action))
        print("in controller it returns:")
        print(env.check_lock_legal(action))
        return env.check_lock_legal(action)

    def play_turn(self, player, observation, info):
        """
        play one player's turn, moving to the next player's turn 
        """
        # if info.farkle: # PLAYER CAN DO ANYTHING< LET STEP HANDLE THIS
        #     # do not update, just return
        reward_this_turn = 0
        player_num = observation["turn"]
        truncated = False
        terminated = False

        while(observation["turn"] == player_num and not truncated and not terminated): # until we move to next player
            print(observation)
            print(info)
            lock, bank = player.play(observation) # prompt current player to play
            print(lock)
            print(bank)
            action = {"lock": lock, "bank": bank}
            observation, reward, terminated, truncated, info = env.step(action)
            reward_this_turn += reward

        # TODO: update player with reward after every action, or after all actions?
        player.update(reward_this_turn)

        return observation, reward, terminated, truncated, info
        
    def play_game(self, seed = None):
        """
        play an entire game
        """
        observation, info = self._new_game(seed)
        print(observation)
        print(info)
        truncated = False
        terminated = False

        # TODO: important for controller not to determine who is next to play. let FarkleEnv and observation tell us who is next to play (in case of b2b farkles)
        while info["winner"] == -1 and not truncated and not terminated: # while game is not over TODO: consider truncated or terminated?
            print(observation)
            print(info)
            current_player = observation["turn"]
            observation, reward, terminated, truncated, info = self.play_turn(self.players[observation["turn"]], observation, info)

        print(f"Winner is player {info["winner"]}!")
        return


if __name__ == "__main__":
    players = [player_testing.ManualPlayer()]
    env = testing.FarkleEnv()
    game = FarkleController(env, players)
    for player in players:
        player.set_controller(game)
    game.play_game()

