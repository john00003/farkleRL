#!/usr/bin/env python3
"""
Test script for the SinglePlayerRLAgent implementation.
"""

import sys
import os

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import farkle as testing
from src import wrapper
from src import controller as controller_testing
from src import player as player_testing
import torch

def test_basic_agent_functionality():
    """Test basic functionality of the RL agent."""
    print("Testing SinglePlayerRLAgent basic functionality...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = testing.FarkleEnv(players=1)
    wrapped_env = wrapper.FarkleEnvSinglePlayerWrapper(env)
    
    # Create RL agent
    agent = player_testing.SinglePlayerRLAgent(device=device, training=True)
    
    # Create controller
    controller = controller_testing.FarkleController(wrapped_env, [agent])
    agent.set_controller(controller)
    
    print("✓ Successfully created environment, agent, and controller")
    
    # Test a few steps
    observation, info = wrapped_env.reset(seed=42)
    print(f"Initial observation keys: {list(observation.keys())}")
    print(f"Dice values: {observation['dice_values']}")
    print(f"Dice locked: {observation['dice_locked']}")
    print(f"Player points: {observation['player_points']}")
    print(f"Points this turn: {observation['points_this_turn']}")
    
    # Test action selection
    try:
        lock_array, bank_action = agent.play(observation)
        print(f"✓ Agent selected action - Lock: {lock_array}, Bank: {bank_action}")
        
        # Test that the action is valid format
        assert isinstance(lock_array, (list, tuple)) or hasattr(lock_array, '__iter__'), "Lock array should be iterable"
        assert isinstance(bank_action, bool), "Bank action should be boolean"
        assert len(lock_array) == len(observation['dice_values']), "Lock array should match dice count"
        
        print("✓ Action format validation passed")
        
    except Exception as e:
        print(f"✗ Error during action selection: {e}")
        return False
    
    # Test update functionality
    try:
        # Simulate receiving a reward
        agent.update(observation, reward=0)
        print("✓ Agent update functionality works")
        
    except Exception as e:
        print(f"✗ Error during agent update: {e}")
        return False
    
    print("✓ All basic functionality tests passed!")
    return True

def test_short_game():
    """Test running a short game with the RL agent using the controller."""
    print("\nTesting short game with RL agent...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create environment
    env = testing.FarkleEnv(players=1)
    wrapped_env = wrapper.FarkleEnvSinglePlayerWrapper(env)
    
    # Create RL agent
    agent = player_testing.SinglePlayerRLAgent(device=device, training=True)
    
    # Create controller
    controller = controller_testing.FarkleController(wrapped_env, [agent])
    agent.set_controller(controller)
    
    try:
        # Use the controller to play a complete turn
        # This is the proper way to test since the controller handles farkles and banking
        print("Running a single turn using the controller...")
        
        # Initialize the game
        observation, info = wrapped_env.reset(seed=123)
        print(f"Initial state - Dice: {observation['dice_values']}, Points: {observation['player_points']}")
        
        # Set initial conditions for play_turn
        reward = -1 if info.get("farkle", False) else 0
        terminated = False
        truncated = False
        
        # Let the controller handle a complete turn
        final_observation, final_reward, final_terminated, final_truncated, final_info, final_curr_player_reward = \
            controller.play_turn(agent, observation, info, reward, terminated, truncated)
        
        print(f"Turn completed!")
        print(f"Final reward: {final_reward}")
        print(f"Game terminated: {final_terminated}")
        print(f"Winner: {final_info.get('winner', -1)}")
        
        if final_info.get('winner', -1) != -1:
            print(f"🎉 Game completed! Winner: {final_info['winner']}")
        else:
            print("Turn completed successfully, game continues...")
        
        print("✓ Successfully completed controller-managed gameplay")
        return True
        
    except Exception as e:
        print(f"✗ Error during short game test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_save_load():
    """Test model saving and loading functionality."""
    print("\nTesting model save/load functionality...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create agent
    agent = player_testing.SinglePlayerRLAgent(device=device, training=True)
    
    try:
        # Test saving
        agent.save_model("test_model.pth")
        print("✓ Model saved successfully")
        
        # Test loading
        agent.load_model("test_model.pth")
        print("✓ Model loaded successfully")
        
        # Clean up
        os.remove("test_model.pth")
        print("✓ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during model save/load test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SinglePlayerRLAgent Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_basic_agent_functionality():
        tests_passed += 1
    
    if test_short_game():
        tests_passed += 1
        
    if test_model_save_load():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The RL agent implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 50)
