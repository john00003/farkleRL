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
    print(f"before helper: {dice_locked}")
    print(dice_values)
    print(string)
    new_locked = [x for x in dice_locked]
    for char in string:
        x = int(char)
        for i, value in enumerate(dice_values): # we find a dice of matching value and undo the lock
            if value == x and new_locked[i] == 0:
                new_locked[i] = 1
                break
    print(f"after helper: {new_locked}")
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
                print(f"before recursive call: {curr_combinations}")
                additional = get_legal_lock_combinations_wrapped(dice_values, _helper_flip_lock(key, dice_values, dice_locked)) #TODO: does this work?
                curr_combinations.extend(additional)
                for combo in curr_combinations:
                    if len(set(combo)) != len(combo):
                        print("meow")
                        print(curr_combinations)
                        print(additional)
                        raise Exception("badness")
                # we get the combinations formed by adding the current combination to the remaining combinations found by recursing
                additional_with_original = copy.deepcopy(additional)
                for combo in additional_with_original:
                    assert unlocked_indices[index:index+len(key)] not in combo
                    combo.extend(unlocked_indices[index:index+len(key)])
                curr_combinations.extend(additional_with_original)
                for combo in curr_combinations:
                    if len(set(combo)) != len(combo):
                        print("woof")
                        print(curr_combinations)
                        print(additional)
                        raise Exception("badness two")
                # sort and do not add duplicates
                for combo in curr_combinations:
                    combo.sort()
                    if combo not in combinations:
                        combinations.append(combo)

    print("final generated combinations:")
    print(combinations)

    return combinations

def check_lock_legal(lock, bank, controller):
    action = {"lock": lock, "bank": bank}
    print(f"action being sent to controller {action}")
    return controller.check_lock_legal(action)

def check_bank_legal(lock, bank, controller):
    action = {"lock": lock, "bank": bank}
    return controller.check_bank_legal(action)

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
    try:
        lock = random.choice(get_legal_lock_combinations(observation))
        lock = convert_lock_indices_to_list(lock, observation)
    except ValueError:
        lock = np.zeros(len(observation["dice_values"]))

    if not check_bank_legal():
        bank = False

    if not check_lock_legal():
        raise Exception("this is probably bad.")

    return lock, bank

class Player:
    def __init__(self):
        self.controller = None
        pass
 
    def set_controller(self, controller):
        self.controller = controller

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


class ManualPlayer(Player):
    def __init__(self):
        pass

    def _get_bank_input(self):
        while True:
            bank = input("bank? (y/N)")
            if bank == "yes" or bank == "y" or bank == "Yes" or bank == "YES" or bank == "Y":
                bank = True
                return bank
            
            # check for good input
            if bank.strip() != "" and bank != "no" and bank != "n" and bank != "No" and bank != "NO" and bank != "N":
                print("Error: Select y/N to bank")
            else:
                bank = False
                return bank

    def _get_lock_input(self, observation):
        while True:
            lock = input("dice to lock?")
            if len(lock) != len(observation["dice_values"]):
                print("Error: string of dice to lock must be of same length as total dice")
                continue

            illegal = False 
            for x in lock:
                if x != "1" and x != "0":
                    print("Error: string of dice to lock must consist of '1' in the places of dice to lock, and '0' in all other places")
                    illegal = True
                    break
            if illegal:
                continue

            return lock

    def _get_action_ensure_legal(self, observation):
        while True:
            lock = self._get_lock_input(observation)
            lock = [int(x) for x in lock]

            bank = self._get_bank_input()

            if not check_bank_legal(lock, bank, self.controller):
                print("Error: illegal to bank")
                continue

            if not check_lock_legal(lock, bank, self.controller):
                print("Error: illegal to lock these dice")
                continue
            
            return lock, bank


    def play(self, observation):
        print(observation)
        lock, bank = self._get_action_ensure_legal(observation)
        return lock, bank

    def update(self, reward):
        # no need to update, this player is not an RL agent
        pass
