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

def choose_random_action(observation):
    """
    this function selects a random action for the agent

    Paramaters
    ----------
    observation: dict
        and observation of the FarkleEnv

    Returns
    -------
    choice: array-like
        the selection of dice to lock, or an empty array if the decision to bank was made or it is not legal to lock anything
    """
    # TODO: this could be bad, if random player had farkled, and coukd not bank, and was prompted to make a turn, they would query random actions forever while checking if they are legal!!!!
    bank = np.random.choice([True, False])

    if not bank:
        try:
            choice = np.random.choice(get_legal_lock_combinations(observation))
        except ValueError:
            return []
    else:
        choice = []
    
    return choice
class RLAgent:

    def __init__(self):
        pass

    def play(self, observation):
        pass

    def update(self, reward):
        pass



class RandomPlayer:

    def __init__(self):
        pass

    def play(self, observation):
        pass

    def update(self, reward):
        pass
