import gymnasium as gym
import numpy as np

# helper functions
def get_legal_lock_combinations(observation):
    """
    this function determines all legal combinations of dice to lock

    Parameters
    ----------
    observation: dict
        a dictionary containing the observations from the FarkleEnv

    Returns
    -------
    combinations: list[list[int]]
        a 2D list containing all the possible selections of dice that can be locked
    """
    unlocked = []
    unlocked_indices = []
    num_unlocked = 0
    for i, (lock, die) in enumerate(zip(dice_locked, dice_values)):
        if not lock:
            unlocked.append(die)
            unlocked_indices.append(i)
            num_unlocked++
    # sort the unlocked dice, but maintain order of indices of those dice
    order = sorted(range(len(unlocked)), key=lambda i: unlocked[i])
    unlocked = [unlocked[i] for i in order]
    unlocked_indices = [unlocked_indices[i] for i in order]
    combinations = []
    string = "".join(unlocked)
    for i in range(1, num_unlocked+1):
        for dict in FarkleEnv.combinations[i]:
            for key in dict.keys():
                index = string.find(key)
                if index == -1: continue
                combinations.append(unlocked_indices[index:index+len(key)+1]) # append the indices that we are allowed to lock

    return combinations

def check_bank_legal(observation):
    """
    checks if a player is allowed to bank

    Parameters
    ----------
    observation: dict
        and observation of the FarkleEnv

    Returns
    -------
    boolean
        a boolean indicating if the player can legally bank
    """
    if observation.player_points + observation.points_this_turn >= 500:
        return True
    return False

def choose_random_action(observation):
    """
    this function selects a random action for the agent

    Paramaters
    ----------
    observation: dict
        and observation of the FarkleEnv

    Returns
    -------
    lock: array-like
        the selection of dice to lock, or an empty array if the decision to bank was made or it is not legal to lock anything
    bank: boolean
        a boolean indicating if the action is to bank
    """
    # TODO: this could be bad, if random player had farkled, and coukd not bank, and was prompted to make a turn, they would query random actions forever while checking if they are legal!!!!
        # where is player being prompted for legal action in while loop?
    # TODO: this method only gets legal combinations to lock, but has no way of checking if it is legal to bank
    bank = np.random.lock([True, False])

    if not bank:
        try:
            lock = np.random.choice(get_legal_lock_combinations(observation))
        except ValueError:
            if check_bank_legal(observation):
                lock = []
                bank = True
            else:                   # no legal actions
                lock = None
                bank = None
    else:
        if check_bank_legal(observation):
            lock = []
        else:
            try:
                lock = np.random.choice(get_legal_lock_combinations(observation))
            except ValueError:      # no legal actions
                lock = None
                bank = None

    return lock, bank


class Player:
    def __init__(self):
        pass

    def play(self, observation):
        """
        gets an action from the player

        Parameters
        ----------
        observation: dict
            and observation of the FarkleEnv

        Returns
        -------
        lock: array-like
            the selection of dice to lock, or an empty array if the decision to bank was made or it is not legal to lock anything, None if there were no legal actions for the player
        bank: boolean
            a boolean indicating if the action is to bank, None if there were no legal actions for the player
        """
        raise NotImplementedError

    def update(self, reward):
        raise NotImplementedError

class RLAgent(Player):
    def __init__(self):
        pass

    def play(self, observation):
        pass

    def update(self, reward):
        pass



class RandomPlayer(Player):
    def __init__(self):
        pass

    def play(self, observation):
        lock, bank = choose_random_action(observation)
        return lock, bank

    def update(self, reward):
        # no need to update, this player is not an RL agent
        pass
