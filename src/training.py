import numpy as np
import time
import matplotlib.pyplot as plt
import os
import torch
from controller_testing import FarkleController
import player_testing
import testing
import wrapper


class FarkleTrainer:
    """
    Training class for the Farkle RL agent that handles multiple games and tracks performance.
    """
    
    def __init__(self, controller):
        """
        Initialize the trainer with a controller.
        
        Parameters
        ----------
        controller : FarkleController
            The game controller that manages gameplay
        """
        self.controller = controller
        self.rl_agent = self._find_rl_agent()
        
    def _find_rl_agent(self):
        """Find the RL agent in the players list."""
        for player in self.controller.players:
            if isinstance(player, player_testing.SinglePlayerRLAgent):
                return player
        raise ValueError("No SinglePlayerRLAgent found in players list!")
    
    def train_agent(self, num_games=1000, save_interval=100, log_interval=50, 
                   model_save_path="farkle_model.pth", plot_results=True, silent_games=True):
        """
        Train the RL agent over multiple games.

        Parameters
        ----------
        num_games : int, optional
            Number of games to play for training (default: 1000)
        save_interval : int, optional
            Save model every N games (default: 100)
        log_interval : int, optional
            Print progress every N games (default: 50)
        model_save_path : str, optional
            Path to save the trained model (default: "farkle_model.pth")
        plot_results : bool, optional
            Whether to plot training results (default: True)
        silent_games : bool, optional
            Whether to suppress game logging during training (default: True)

        Returns
        -------
        training_stats : dict
            Dictionary containing training statistics
        """
        print(f"Starting training for {num_games} games...")
        print(f"Model will be saved every {save_interval} games to {model_save_path}")
        
        # Training statistics
        turns_per_game = []
        rewards_per_game = []
        games_won = 0
        start_time = time.time()
        
        # Temporarily disable controller logging if silent mode
        original_log = self.controller.log
        if silent_games:
            self.controller.log = lambda x: None
        
        # Training loop
        for game_num in range(1, num_games + 1):
            # Play a game
            result = self._play_training_game()
            
            # Track statistics
            turns_per_game.append(result["turns"])
            rewards_per_game.append(result["total_reward"])
            
            if result["winner"] == 0:  # Assuming single player, winner is always 0
                games_won += 1
            
            # Progress logging
            if game_num % log_interval == 0:
                avg_turns = np.mean(turns_per_game[-log_interval:])
                avg_reward = np.mean(rewards_per_game[-log_interval:])
                elapsed_time = time.time() - start_time
                
                print(f"Game {game_num}/{num_games} | "
                      f"Avg Turns (last {log_interval}): {avg_turns:.1f} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # Save model at intervals
            if game_num % save_interval == 0:
                checkpoint_path = f"{model_save_path}.checkpoint_{game_num}"
                self.rl_agent.save_model(checkpoint_path)
                print(f"Model checkpoint saved: {checkpoint_path}")
        
        # Restore original logging
        self.controller.log = original_log
        
        # Final model save
        self.rl_agent.save_model(model_save_path)
        print(f"Final model saved: {model_save_path}")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_turns_all = np.mean(turns_per_game)
        avg_reward_all = np.mean(rewards_per_game)
        win_rate = games_won / num_games
        
        training_stats = {
            "num_games": num_games,
            "total_time": total_time,
            "avg_turns_per_game": avg_turns_all,
            "avg_reward_per_game": avg_reward_all,
            "win_rate": win_rate,
            "turns_per_game": turns_per_game,
            "rewards_per_game": rewards_per_game
        }
        
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"{'='*50}")
        print(f"Games played: {num_games}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average turns per game: {avg_turns_all:.1f}")
        print(f"Average reward per game: {avg_reward_all:.2f}")
        print(f"Games won: {games_won}/{num_games} ({win_rate:.1%})")
        print(f"Final model saved to: {model_save_path}")
        
        # Plot results if requested
        if plot_results:
            self.plot_training_results(training_stats)
        
        return training_stats
    
    def _play_training_game(self):
        """
        Play a single training game and return results.
        
        Returns
        -------
        result : dict
            Dictionary containing game results
        """
        observation, info = self.controller._new_game()
        truncated = False
        terminated = False
        reward = -1 if info["farkle"] else 0
        turns = 0
        total_reward = 0

        while info["winner"] == -1 and not truncated and not terminated:
            current_player = observation["turn"]
            observation, reward, terminated, truncated, info = self.controller.play_turn(
                self.controller.players[observation["turn"]], observation, info, reward, terminated, truncated
            )
            total_reward += reward
            turns += 1

        return {
            "winner": info["winner"],
            "turns": turns,
            "total_reward": total_reward
        }

    def plot_training_results(self, training_stats, save_plot=True):
        """
        Plot training results showing performance over time.

        Parameters
        ----------
        training_stats : dict
            Training statistics from train_agent
        save_plot : bool, optional
            Whether to save the plot to a file (default: True)
        """
        turns_per_game = training_stats["turns_per_game"]
        rewards_per_game = training_stats["rewards_per_game"]
        
        # Create moving averages for smoother plots
        window_size = max(10, len(turns_per_game) // 50)  # Adaptive window size
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window), 'valid') / window
        
        turns_ma = moving_average(turns_per_game, window_size)
        rewards_ma = moving_average(rewards_per_game, window_size)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot turns per game
        ax1.plot(turns_per_game, alpha=0.3, color='blue', label='Raw data')
        ax1.plot(range(window_size-1, len(turns_per_game)), turns_ma, 
                color='red', linewidth=2, label=f'Moving average ({window_size} games)')
        ax1.set_title('Average Turns per Game During Training')
        ax1.set_xlabel('Game Number')
        ax1.set_ylabel('Turns to Win')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot rewards per game
        ax2.plot(rewards_per_game, alpha=0.3, color='green', label='Raw data')
        ax2.plot(range(window_size-1, len(rewards_per_game)), rewards_ma, 
                color='orange', linewidth=2, label=f'Moving average ({window_size} games)')
        ax2.set_title('Average Reward per Game During Training')
        ax2.set_xlabel('Game Number')
        ax2.set_ylabel('Total Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = 'farkle_training_results.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Training results plot saved: {plot_filename}")
        
        plt.show()

    def evaluate_agent(self, num_games=100, model_path=None):
        """
        Evaluate the trained agent's performance.

        Parameters
        ----------
        num_games : int, optional
            Number of games to play for evaluation (default: 100)
        model_path : str, optional
            Path to load model from (if None, uses current model)

        Returns
        -------
        eval_stats : dict
            Dictionary containing evaluation statistics
        """
        print(f"Evaluating agent over {num_games} games...")
        
        # Load model if specified
        if model_path and os.path.exists(model_path):
            self.rl_agent.load_model(model_path)
            print(f"Loaded model from {model_path}")
        
        # Set agent to evaluation mode (no exploration)
        self.rl_agent.set_training_mode(False)
        
        # Disable logging during evaluation
        original_log = self.controller.log
        self.controller.log = lambda x: None
        
        # Run evaluation games
        turns_list = []
        rewards_list = []
        games_won = 0
        
        for game_num in range(num_games):
            result = self._play_training_game()
            turns_list.append(result["turns"])
            rewards_list.append(result["total_reward"])
            
            if result["winner"] == 0:
                games_won += 1
        
        # Restore logging
        self.controller.log = original_log
        
        # Calculate statistics
        eval_stats = {
            "num_games": num_games,
            "avg_turns": np.mean(turns_list),
            "std_turns": np.std(turns_list),
            "min_turns": np.min(turns_list),
            "max_turns": np.max(turns_list),
            "avg_reward": np.mean(rewards_list),
            "std_reward": np.std(rewards_list),
            "win_rate": games_won / num_games,
            "turns_list": turns_list,
            "rewards_list": rewards_list
        }
        
        print(f"\n{'='*40}")
        print(f"Evaluation Results")
        print(f"{'='*40}")
        print(f"Games played: {num_games}")
        print(f"Average turns: {eval_stats['avg_turns']:.1f} ± {eval_stats['std_turns']:.1f}")
        print(f"Turn range: {eval_stats['min_turns']}-{eval_stats['max_turns']}")
        print(f"Average reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"Win rate: {eval_stats['win_rate']:.1%}")
        
        # Set agent back to training mode
        self.rl_agent.set_training_mode(True)
        
        return eval_stats


def setup_training_environment():
    """
    Set up the training environment with a SinglePlayerRLAgent.
    
    Returns
    -------
    trainer : FarkleTrainer
        Configured trainer ready for training
    """
    # Create RL agent
    rl_agent = player_testing.SinglePlayerRLAgent(training=True)
    
    # Create environment and controller
    env = testing.FarkleEnv(players=1)
    wrapped_env = wrapper.FarkleEnvSinglePlayerWrapper(env)
    controller = FarkleController(wrapped_env, [rl_agent])
    
    # Set controller for agent
    rl_agent.set_controller(controller)
    
    # Create trainer
    trainer = FarkleTrainer(controller)
    
    return trainer


def main():
    """
    Main training script that demonstrates how to train the Farkle RL agent.
    """
    print("Setting up Farkle RL training environment...")
    
    # Setup training
    trainer = setup_training_environment()
    
    print(f"Training agent device: {trainer.rl_agent.device}")
    print("Starting training...")
    
    # Train the agent
    training_stats = trainer.train_agent(
        num_games=500,  # Start with smaller number for testing
        save_interval=100,
        log_interval=25,
        model_save_path="farkle_rl_model.pth",
        plot_results=True,
        silent_games=True
    )
    
    print("\nTraining completed!")
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_stats = trainer.evaluate_agent(num_games=50)
    
    return training_stats, eval_stats


if __name__ == "__main__":
    # Run training
    training_stats, eval_stats = main()
