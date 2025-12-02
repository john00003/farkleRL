import gymnasium as gym
import numpy as np
import random
import copy
import torch
from .farkle import FarkleEnv, get_legal_lock_combinations
from .dqn_agent import (
    DQNAgent, 
    select_action_with_network, 
    validate_and_convert_action, 
    store_transition_and_train
)

# helper functions

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
    Simplified to use helper functions from dqn_agent.py
    """
    
    def __init__(self, agent=None, device='cpu', training=True):
        super().__init__()
        self.device = torch.device(device)
        self.training = training
        # Decoupled: Accept an agent instance or create one if not provided
        if agent is None:
            self.agent = DQNAgent(device=self.device)
        else:
            self.agent = agent
        
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
            select_action_with_network(self.agent, observation, get_legal_lock_combinations, self.training)
        
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

        self.log(f"RL agent currently has {observation["player_points"]} points")
        
        return lock_array, bank_action
    
    def update(self, observation, reward):
        """
        Update the neural network with the received reward using helper function.
        """
        if not self.training or self.last_state is None:
            return
        
        # Use helper function to store transition and train
        store_transition_and_train(
            self.agent, self.last_state, self.last_legal_mask, 
            self.last_bank_action, self.last_lock_action, reward,
            observation, get_legal_lock_combinations, self.training
        )
        
        self.log(f"Updated with reward: {reward}")
    
    def save_model(self, filepath):
        torch.save(self.agent.policy_net.state_dict(), filepath)
        self.log(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.agent.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
        self.log(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training=True):
        self.training = training
        if training:
            self.agent.policy_net.train()
        else:
            self.agent.policy_net.eval()



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
