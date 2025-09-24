import gymnasium as gym
import numpy as np
import random
import copy
from testing import FarkleEnv
import torch
from single_player_neural_network import (
    FarkleTrainer, 
    select_action_with_network, 
    validate_and_convert_action, 
    store_transition_and_train
)

# helper functions
def get_legal_lock_combinations(observation):
    """
    Determine all legal combinations of dice to lock, given the current state.

    Parameters
    ----------
    observation : dict
        Observation from the Farkle environment. Must contain:
            - "dice_locked": list[int]
                1 if the die is currently locked, 0 otherwise
            - "dice_values": list[int]
                current values rolled for each die

    Returns
    -------
    combinations : list[list[int]]
        A list of possible dice index selections that may be locked.
        Each inner list contains indices (into dice_values) of dice
        that can be locked together. 

        NOTE: This is not in the action format expected by FarkleEnv;
        it must be converted before being passed as an action.
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
        0 if the die was previously unlocked, but we are locking it, 1 otherwise
    """
    new_locked = [x for x in dice_locked]
    for char in string:
        x = int(char)
        for i, value in enumerate(dice_values): # we find a dice of matching value and undo the lock
            if value == x and new_locked[i] == 0:
                new_locked[i] = 1
                break
    return new_locked

def get_legal_lock_combinations_wrapped(dice_values, dice_locked):
    """
    Recursive helper to enumerate all possible legal lock combinations.

    Parameters
    ----------
    dice_values : list[int]
        Values of the dice currently rolled.
    dice_locked : list[int]
        1 if the die is already locked, 0 otherwise.

    Returns
    -------
    combinations : list[list[int]]
        All possible index sets of dice that may be locked,
        constructed recursively from valid scoring subsets.
    """
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
                for combo in curr_combinations:
                    if len(set(combo)) != len(combo):
                        raise Exception("badness")
                # we get the combinations formed by adding the current combination to the remaining combinations found by recursing
                additional_with_original = copy.deepcopy(additional)
                for combo in additional_with_original:
                    assert unlocked_indices[index:index+len(key)] not in combo
                    combo.extend(unlocked_indices[index:index+len(key)])
                curr_combinations.extend(additional_with_original)
                for combo in curr_combinations:
                    if len(set(combo)) != len(combo):
                        raise Exception("badness two")
                # sort and do not add duplicates
                for combo in curr_combinations:
                    combo.sort()
                    if combo not in combinations:
                        combinations.append(combo)

    return combinations

def check_lock_legal(lock, bank, controller):
    action = {"lock": lock, "bank": bank}
    return controller.check_lock_legal(action)

def check_bank_legal(lock, bank, controller):
    action = {"lock": lock, "bank": bank}
    return controller.check_bank_legal(action)

def convert_lock_indices_to_list(indices, observation):
    lock = np.zeros(len(observation["dice_values"]))
    lock[indices] = 1
    return lock

def choose_random_action(observation, controller):
    """
    Select a random action (lock and/or bank) for the player.

    Parameters
    ----------
    observation : dict
        Observation of the Farkle environment.
    controller : object
        The controller that enforces rules. Must provide:
            - check_lock_legal(action: dict) -> bool
            - check_bank_legal(action: dict) -> bool

    Returns
    -------
    lock : np.ndarray
        Binary array of length equal to number of dice.
        1 indicates the die is locked, 0 otherwise.
    bank : bool
        True if the action is to bank, False otherwise.

    Raises
    ------
    Exception
        If no legal lock action could be chosen.
    """
    # Get all possible legal lock combinations
    possible_actions = []
    for lock_combo in get_legal_lock_combinations(observation):
        lock = convert_lock_indices_to_list(lock_combo, observation)
        possible_actions.append((False, lock))
        if check_bank_legal(lock, True, controller):
            possible_actions.append((True, lock))

    # Add empty lock with banking option if legal
    empty_lock = np.zeros(len(observation["dice_values"]))
    if check_bank_legal(empty_lock, True, controller):
        possible_actions.append((True, empty_lock))

    if not possible_actions:
        raise Exception("No legal actions available.")

    bank, lock = random.choice(possible_actions)

    if not check_lock_legal(lock, bank, controller):
        raise Exception("Selected illegal action.")

    return lock, bank

class Player:
    def __init__(self):
        self.controller = None
        pass

    def log(self, string):
        print(f"PLAYER: {string}.")
 
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

    def update(self, observation, reward):
        raise NotImplementedError

class RLAgent(Player):
    def __init__(self):
        pass

    def play(self, observation):
        pass

    def update(self, observation, reward):
        pass



class RandomPlayer(Player):
    def __init__(self):
        pass

    def play(self, observation):
        self.log("Getting random action...")
        lock, bank = choose_random_action(observation, self.controller)
        if bank:
            self.log(f"Random player decided to bank, and lock {lock}")
        else:
            self.log(f"Random player decided to lock {lock}")
        return lock, bank

    def update(self, observation, reward):
        # no need to update, this player is not an RL agent
        pass


class SinglePlayerRLAgent(Player):
    """
    Reinforcement Learning agent for single-player Farkle using neural networks.
    Simplified to use helper functions from single_player_neural_network.py
    """
    
    def __init__(self, device='cpu', training=True):
        super().__init__()
        self.device = torch.device(device)
        self.training = training
        self.trainer = FarkleTrainer(device=self.device)
        
        # State tracking for experience replay
        self.last_state = None
        self.last_legal_mask = None
        self.last_bank_action = None
        self.last_lock_action = None
        
    def log(self, string):
        if self.training:
            print(f"RL AGENT: {string}")
    
    def play(self, observation):
        """
        Select an action using the neural network with helper functions.
        """
        # Use helper function to select action with network
        bank_action, lock_action_idx, chosen_combination, state, legal_mask, legal_combinations = \
            select_action_with_network(self.trainer, observation, get_legal_lock_combinations, self.training)
        
        # Use helper function to validate and convert action
        lock_array, bank_action = validate_and_convert_action(
            chosen_combination, bank_action, observation, 
            convert_lock_indices_to_list, check_lock_legal, self.controller
        )
        
        # Store current state and action for next update
        self.last_state = state
        self.last_legal_mask = legal_mask
        self.last_bank_action = bank_action
        self.last_lock_action = lock_action_idx
        
        if bank_action:
            self.log(f"RL agent decided to bank with lock combination: {chosen_combination}")
        else:
            self.log(f"RL agent decided to lock dice: {chosen_combination}")
        
        return lock_array, bank_action
    
    def update(self, observation, reward):
        """
        Update the neural network with the received reward using helper function.
        """
        if not self.training or self.last_state is None:
            return
        
        # Use helper function to store transition and train
        store_transition_and_train(
            self.trainer, self.last_state, self.last_legal_mask, 
            self.last_bank_action, self.last_lock_action, reward,
            observation, get_legal_lock_combinations, self.training
        )
        
        self.log(f"Updated with reward: {reward}")
    
    def save_model(self, filepath):
        torch.save(self.trainer.policy_net.state_dict(), filepath)
        self.log(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.trainer.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.trainer.target_net.load_state_dict(self.trainer.policy_net.state_dict())
        self.log(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training=True):
        self.training = training
        if training:
            self.trainer.policy_net.train()
        else:
            self.trainer.policy_net.eval()


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
                self.log("Error: Select y/N to bank")
            else:
                bank = False
                return bank

    def _get_lock_input(self, observation):
        """
        Prompt the user to enter dice to lock.

        The user must input a string of '0' and '1' with length equal
        to the number of dice. '1' means the die at that position is locked.

        Parameters
        ----------
        observation : dict
            Current environment observation.

        Returns
        -------
        lock : str
            String of '0' and '1' indicating locked dice.
        """
        while True:
            lock = input("dice to lock?")
            if len(lock) != len(observation["dice_values"]):
                self.log("Error: string of dice to lock must be of same length as total dice")
                continue

            illegal = False 
            for x in lock:
                if x != "1" and x != "0":
                    self.log("Error: string of dice to lock must consist of '1' in the places of dice to lock, and '0' in all other places")
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
                self.log("Error: illegal to bank")
                continue

            if not check_lock_legal(lock, bank, self.controller):
                self.log("Error: illegal to lock these dice")
                continue
            
            return lock, bank


    def play(self, observation):
        lock, bank = self._get_action_ensure_legal(observation)
        return lock, bank

    def update(self, observation, reward):
        # no need to update, this player is not an RL agent
        pass
