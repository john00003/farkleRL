import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
import random
import math

class FarkleNet(nn.Module):
    """
    Neural network for playing Farkle.
    
    The network takes the game state as input and outputs:
    1. A probability of banking (single sigmoid output)
    2. Logits for each possible lock combination (will be masked for legal actions)
    """
    
    def __init__(self, num_dice=6, max_points=10000, max_lock_combinations=64):
        super(FarkleNet, self).__init__()
        
        self.num_dice = num_dice
        self.max_points = max_points
        self.max_lock_combinations = max_lock_combinations
        
        # - dice_values: num_dice (values 1-6)
        # - dice_locked: num_dice (binary)
        # - player_points: 1 (normalized)
        # - points_this_turn: 1 (normalized)
        input_size = num_dice + num_dice + 1 + 1
        
        # layers shared between the two heads
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # bank decision head
        self.bank_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # lock combination head
        self.lock_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_lock_combinations)
        )
    
    def forward(self, state, legal_lock_mask=None):
        """
        Forward pass through the network.
        
        Args:
            state: Tensor of shape (batch_size, input_size) containing the game state
            legal_lock_mask: Tensor of shape (batch_size, max_lock_combinations) 
                           where 1 indicates legal actions and 0 indicates illegal ones
        
        Returns:
            bank_prob: Tensor of shape (batch_size, 1) with banking probabilities
            lock_logits: Tensor of shape (batch_size, max_lock_combinations) with lock logits
        """
        shared_features = self.shared_layers(state)
        
        # bank decision
        bank_prob = self.bank_head(shared_features)
        
        # lock combination logits
        lock_logits = self.lock_head(shared_features)
        
        if legal_lock_mask is not None:
            # illegal actions are very negative values
            lock_logits = lock_logits.masked_fill(legal_lock_mask == 0, -1e9)
        
        return bank_prob, lock_logits
    
    def get_action_probabilities(self, state, legal_lock_mask):
        """
        Get action probabilities for the given state.
        
        Returns:
            bank_prob: Probability of banking
            lock_probs: Probability distribution over legal lock combinations
        """
        bank_prob, lock_logits = self.forward(state, legal_lock_mask)
        lock_probs = F.softmax(lock_logits, dim=-1)
        return bank_prob, lock_probs


def prepare_state_tensor(observation, device='cpu'):
    """
    Convert a Farkle observation to a tensor suitable for the neural network.
    
    Args:
        observation: Dictionary containing the game state
        device: PyTorch device to place the tensor on
    
    Returns:
        Tensor of shape (input_size,) ready for the network
    """
    # normalize dice values to [0, 1]
    dice_values = torch.tensor(observation["dice_values"], dtype=torch.float32) - 1.0
    dice_values = dice_values / 5.0
    
    dice_locked = torch.tensor(observation["dice_locked"], dtype=torch.float32)
    
    # normalize player points
    player_points = torch.tensor([observation["player_points"]], dtype=torch.float32) / 10000.0
    
    # normalize points this turn
    points_this_turn = torch.tensor([observation["points_this_turn"]], dtype=torch.float32) / 10000.0
    
    state = torch.cat([dice_values, dice_locked, player_points, points_this_turn])
    
    return state.to(device)


def prepare_legal_mask(legal_combinations, max_combinations=64, device='cpu'):
    """
    Convert legal lock combinations to a mask tensor.
    
    Args:
        legal_combinations: List of legal lock combination indices
        max_combinations: Maximum number of combinations supported
        device: PyTorch device to place the tensor on
    
    Returns:
        Tensor of shape (max_combinations,) with 1s for legal actions and 0s for illegal
    """
    mask = torch.zeros(max_combinations, dtype=torch.float32)
    
    # Mark legal combinations as 1
    for i in legal_combinations:
        if i < max_combinations:
            mask[i] = 1.0
    
    return mask.to(device)


# we define a state, action, next action transition
Transition = namedtuple('Transition', ('state', 'legal_mask', 'bank_action', 'lock_action', 'reward', 'next_state', 'next_legal_mask', 'done'))

class ReplayMemory:
    """Experience replay buffer for storing transitions."""
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class FarkleTrainer:
    """
    Trainer class for the Farkle neural network using DQN-style learning.
    """
    
    def __init__(self, device='cpu', learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=2000, memory_capacity=10000):
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # define a policy and target network
        self.policy_net = FarkleNet().to(device)
        self.target_net = FarkleNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Adam optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_capacity)
        
        # loss functions
        self.bank_criterion = nn.BCELoss()
        self.lock_criterion = nn.CrossEntropyLoss()
    
    def select_action(self, state, legal_lock_mask, legal_combinations):
        """
        Select an action using epsilon-greedy strategy.
        
        Returns:
            bank_action: Boolean indicating whether to bank
            lock_action: Index of the chosen lock combination
        """
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            # exploitation
            with torch.no_grad():
                state_batch = state.unsqueeze(0)
                legal_mask_batch = legal_lock_mask.unsqueeze(0)
                
                bank_prob, lock_probs = self.policy_net.get_action_probabilities(state_batch, legal_mask_batch)
                
                # sample action from output probabilities
                bank_action = torch.bernoulli(bank_prob).item() > 0.5
                lock_action = torch.multinomial(lock_probs, 1).item()
        else:
            # exploration
            bank_action = random.choice([True, False])
            lock_action = random.choice(range(len(legal_combinations)))
        
        return bank_action, lock_action
    
    def optimize_model(self, batch_size=32):
        """Perform one step of optimization on the policy network."""
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # convert to tensors
        state_batch = torch.stack(batch.state)
        legal_mask_batch = torch.stack(batch.legal_mask)
        bank_action_batch = torch.tensor(batch.bank_action, dtype=torch.float32).unsqueeze(1)
        lock_action_batch = torch.tensor(batch.lock_action, dtype=torch.long)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        
        # compute current Q values
        bank_probs, lock_logits = self.policy_net(state_batch, legal_mask_batch)
        
        # apply loss functions
        bank_loss = self.bank_criterion(bank_probs, bank_action_batch)
        lock_loss = self.lock_criterion(lock_logits, lock_action_batch)
        total_loss = bank_loss + lock_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def update_target_network(self, tau=0.005):
        """Soft update of the target network."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)


# Helper functions for action selection and processing
def prepare_legal_combinations_with_indices(legal_combinations):
    """
    Process legal lock combinations and create index mapping.
    
    Args:
        legal_combinations: List of legal lock combinations (as dice indices)
        
    Returns:
        legal_combinations: Processed list with empty combination added if needed
        legal_indices: List of indices corresponding to each legal combination
    """
    # TODO: validate
    # Validate no invalid combinations
    if [0,0,0,0,0,0] in legal_combinations:
        raise Exception("Invalid legal combination detected")
    
    # Always add the empty combination (no dice locked) as index 0
    # TODO: validate
    if [] not in legal_combinations:
        legal_combinations.insert(0, [])
    
    # Create indices for each legal combination
    legal_indices = list(range(len(legal_combinations)))
    
    return legal_combinations, legal_indices


def select_action_with_network(trainer, observation, legal_combinations_fn, training=True):
    """
    Select an action using the neural network with proper input preparation and masking.
    
    Args:
        trainer: FarkleTrainer instance
        observation: Game observation dictionary
        legal_combinations_fn: Function to get legal combinations from observation
        training: Whether to use training mode (epsilon-greedy) or evaluation mode (greedy)
        
    Returns:
        bank_action: Boolean indicating whether to bank
        lock_action_idx: Index of chosen lock combination
        chosen_combination: The actual dice indices to lock
        state: Prepared state tensor (for storing in replay buffer)
        legal_mask: Legal action mask tensor (for storing in replay buffer)
        legal_combinations: List of legal combinations (for validation)
    """
    legal_combinations = legal_combinations_fn(observation)
    legal_combinations, legal_indices = prepare_legal_combinations_with_indices(legal_combinations)
    
    if len(legal_combinations) == 0:
        raise Exception("No legal lock combinations available.")
    
    state = prepare_state_tensor(observation, trainer.device)
    
    legal_mask = prepare_legal_mask(legal_indices, device=trainer.device)
    
    if training:
        bank_action, lock_action_idx = trainer.select_action(state, legal_mask, legal_combinations)
    else:
        # if not training, we are evaluations
        # use greedy selection, no exploration
        with torch.no_grad():
            state_batch = state.unsqueeze(0)
            legal_mask_batch = legal_mask.unsqueeze(0)
            
            bank_prob, lock_probs = trainer.policy_net.get_action_probabilities(state_batch, legal_mask_batch)
            
            bank_action = bank_prob.item() > 0.5
            lock_action_idx = torch.argmax(lock_probs, dim=-1).item()
    
    # convert lock action index to actual dice combination
    if lock_action_idx < len(legal_combinations):
        chosen_combination = legal_combinations[lock_action_idx]
    else:
        raise Exception("Lock action index out of bounds!")
    
    return bank_action, lock_action_idx, chosen_combination, state, legal_mask, legal_combinations


def validate_and_convert_action(chosen_combination, bank_action, observation, convert_lock_indices_fn, check_lock_legal_fn, controller):
    """
    Validate the chosen action and convert it to the proper format.
    
    Args:
        chosen_combination: List of dice indices to lock
        bank_action: Boolean indicating whether to bank
        observation: Game observation
        convert_lock_indices_fn: Function to convert indices to lock array
        check_lock_legal_fn: Function to check if action is legal
        controller: Game controller for validation
        
    Returns:
        lock_array: Binary array indicating which dice to lock
        bank_action: Validated bank action (may be forced to False if banking is illegal)
    """
    # convert to lock array format
    lock_array = convert_lock_indices_fn(chosen_combination, observation)
    
    if not check_lock_legal_fn(lock_array, False, controller):  # consider no banking, since choice to bank can be adjusted easily
        # exception must be raised if lock is not legal - mask must be wrong.
        raise Exception("Selected illegal lock combination!")
    
    if bank_action:
        if not controller.check_bank_legal({"lock": lock_array, "bank": True}):
            # Banking is illegal, force bank_action to False
            bank_action = False
    
    return lock_array, bank_action


def store_transition_and_train(trainer, state, legal_mask, bank_action, lock_action_idx, reward, 
                              next_observation, legal_combinations_fn, training=True):
    """
    Store the transition in replay memory and perform training step.
    
    Args:
        trainer: FarkleTrainer instance
        state: Current state tensor
        legal_mask: Current legal action mask
        bank_action: Action taken (bank)
        lock_action_idx: Action taken (lock combination index)
        reward: Reward received
        next_observation: Next observation from environment
        legal_combinations_fn: Function to get legal combinations
        training: Whether to perform training
    """
    if not training:
        return
    
    # prepare next state if turn not done
    done = reward != 0
    
    if not done and next_observation is not None:
        next_legal_combinations = legal_combinations_fn(next_observation)
        next_legal_combinations, next_legal_indices = prepare_legal_combinations_with_indices(next_legal_combinations)
        next_state = prepare_state_tensor(next_observation, trainer.device)
        next_legal_mask = prepare_legal_mask(next_legal_indices, device=trainer.device)
    else:
        next_state = None
        next_legal_mask = None
    
    trainer.memory.push(
        state,
        legal_mask,
        bank_action,
        lock_action_idx,
        reward,
        next_state,
        next_legal_mask,
        done
    )
    
    trainer.optimize_model()
    
    # update target network periodically
    if trainer.steps_done % 100 == 0:
        trainer.update_target_network()
