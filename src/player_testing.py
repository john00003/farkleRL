import gymnasium as gym
import numpy as np
import random
import copy
from testing import FarkleEnv

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
        each list contains indices of the dice to locked
            NOTE: this is not the format of the lock action the FarkleEnv expects. it must be converted.
    """
    dice_locked = observation["dice_locked"]
    dice_values = observation["dice_values"]

    return get_legal_lock_combinations_wrapped(dice_values, dice_locked)

def _helper_flip_lock(string, dice_values, dice_locked):
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
            if value == x and dice_locked[i] == 0:
                new_locked[i] = 1
                break
    return new_locked

def get_legal_lock_combinations_wrapped(dice_values, dice_locked):
    unlocked = []
    unlocked_indices = []
    num_unlocked = 0
    for i, (lock, die) in enumerate(zip(dice_locked, dice_values)):
        if not lock:
            unlocked.append(die)
            unlocked_indices.append(i)
            num_unlocked += 1
    # sort the unlocked dice, but maintain order of indices of those dice
    order = sorted(range(len(unlocked)), key=lambda i: unlocked[i])
    unlocked = [str(unlocked[i]) for i in order]
    unlocked_indices = [unlocked_indices[i] for i in order]
    combinations = []
    string = "".join(unlocked)
    for i in range(1, num_unlocked+1):
        for dict in FarkleEnv.combinations[i]:
            for key in dict.keys():
                index = string.find(key)
                if index == -1: continue
                curr_combinations = []
                curr_combinations.append(unlocked_indices[index:index+len(key)]) # append the indices that we are allowed to lock
                # we get the possible combinations of dice to lock without the dice that we locked in the current recursion level
                additional = get_legal_lock_combinations_wrapped(dice_values, _helper_flip_lock(key, dice_values, dice_locked)) #TODO: does this work?
                curr_combinations.extend(additional)
                # we get the combinations formed by adding the current combination to the remaining combinations found by recursing
                additional_with_original = copy.deepcopy(additional)
                for combo in additional_with_original:
                    combo.extend(unlocked_indices[index:index+len(key)])
                curr_combinations.extend(additional_with_original)
                # sort and do not add duplicates
                for combo in curr_combinations:
                    combo.sort()
                    if combo not in combinations:
                        combinations.append(combo)

    print("final generated combinations:")
    print(combinations)

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
    if observation["player_points"] + observation["points_this_turn"] >= 500:
        return True
    return False

def convert_lock_indices_to_list(indices, observation):
    lock = np.zeros(len(observation["dice_values"]))
    lock[indices] = 1
    return lock

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
    # TODO: check if we still should expect to get ValueError
    bank = np.random.choice([True, False])

    if not bank:
        try:
            lock = random.choice(get_legal_lock_combinations(observation))
        except ValueError:
            if check_bank_legal(observation):
                lock = []
                bank = True
            else:                   # no legal actions
                print("tried to lock some dice, did not succeed, then bank was illegal")
                lock = None
                bank = None
    else:
        if check_bank_legal(observation):
            lock = []
        else:
            bank = False
            try:
                lock = random.choice(get_legal_lock_combinations(observation))
            except ValueError:      # no legal actions
                print("tried to bank, did not succeed, then no dice to lock")
                lock = None
                bank = None

    # convert to the format that FarkleEnv expects
    lock = convert_lock_indices_to_list(lock, observation)
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
        print("player is selecting:")
        print(lock)
        print(bank)
        return lock, bank

    def update(self, reward):
        # no need to update, this player is not an RL agent
        pass
