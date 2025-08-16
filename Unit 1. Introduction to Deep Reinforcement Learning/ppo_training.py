import subprocess
# Set your token as an environment variable
# This command will use the token from the environment variable
subprocess.run(['apt', 'install', 'swig', 'cmake'], check=True)
subprocess.run(['pip', 'install', '-r', 'https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt'], check=True)
# Update package lists
subprocess.run(['sudo', 'apt-get', 'update'], check=True)
# Install python3-opengl
subprocess.run(['sudo', 'apt-get', 'install', '-y', 'python3-opengl'], check=True)
# Install ffmpeg
subprocess.run(['sudo', 'apt', 'install', 'ffmpeg'], check=True)
# Install xvfb
subprocess.run(['sudo', 'apt', 'install', 'xvfb'], check=True)
# Install pyvirtualdisplay using pip
subprocess.run(['pip3', 'install', 'pyvirtualdisplay'], check=True)


import os
from os.path import join as pjoin
from dotenv import load_dotenv
load_dotenv('/content/drive/MyDrive/.env')

import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
assert torch.cuda.is_available()

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login 
import gymnasium
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor


assert os.environ.get('HF_TOKEN') is not None
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get('HF_TOKEN')
assert os.environ["HUGGING_FACE_HUB_TOKEN"] is not None
# subprocess.run(["huggingface-cli", "login"])
subprocess.run(["hf", "auth", "login"])

### nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1


class TrainingLoggerCallback(BaseCallback):
    """
    A callback to log all training metrics from the model's logger history.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None

    def _on_training_start(self) -> None:
        # This will save all the logger's history to a CSV file
        self.model.set_logger(configure(folder=self.model.logger.dir, format_strings=['csv']))
        self.pbar = tqdm(total=self.locals['total_timesteps'], desc="Training")

    def _on_step(self) -> bool:
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        """Close the progress bar."""
        self.pbar.close()

# Create the environment
env = make_vec_env('LunarLander-v2', n_envs=16)

# Define the PPO architecture
model_name = "ppo-LunarLander-v2"
model = PPO(
  policy = 'MlpPolicy',
  env = env,
  n_steps = 1024,
  batch_size = 64,
  n_epochs = 10,
  gamma = 0.999,
  gae_lambda = 0.98,
  ent_coef = 0.01,
  verbose=1
)

# Use the logger callback to save all metrics
my_logger_callback = TrainingLoggerCallback()
# Train for enough steps to get multiple log entries
model.learn(total_timesteps=int(1024 * 16 * 60), callback=my_logger_callback)
model.save(model_name)

# --- After Training ---
# Read the saved CSV log file to get the training metrics
csv_path = model.logger.dir + '/progress.csv'
print(f"csv_path: {csv_path}")
if os.path.exists(csv_path):
    import pandas as pd
    log_df = pd.read_csv(csv_path)

    # Plot the policy loss
    plt.figure(figsize=(16, 5))
    plt.plot(log_df['time/iterations'], log_df['train/loss'], color='dodgerblue')
    plt.scatter(log_df['time/iterations'], log_df['train/loss'], color='dodgerblue', s=15);
    plt.title("PPO Policy Loss During Training")
    plt.xlabel("iterations")
    plt.ylabel("Train Loss")
    plt.grid(True)
    plt.savefig(pjoin(os.getcwd(), 'train_loss.png'), bbox_inches='tight')
    plt.show()

    # Plot the mean reward
    plt.figure(figsize=(16, 5))
    plt.plot(log_df['time/iterations'], log_df['rollout/ep_rew_mean'], color='dodgerblue')
    plt.scatter(log_df['time/iterations'], log_df['rollout/ep_rew_mean'], color='dodgerblue', s=15);
    plt.title("PPO Mean Episode Reward During Training")
    plt.xlabel("iterations")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True)
    plt.savefig(pjoin(os.getcwd(), 'mean_reward.png'), bbox_inches='tight')
    plt.show()

    print("Train loss:", log_df['train/loss'].tolist())
    print("Mean Reward Values:", log_df['rollout/ep_rew_mean'].tolist())
else:
    print("No log file found.")


## TODO: Define a repo_id
## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
repo_id = 'antoncio/lunar_lander'

# TODO: Define the name of the environment
env_id = 'LunarLander-v2'

# Create the evaluation env and set the render_mode="rgb_array"
eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])


# TODO: Define the model architecture we used
model_architecture = "PPO MLP"

## TODO: Define the commit message
commit_message = "init commit"

# method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
package_to_hub(model=model, # Our trained model
               model_name=model_name, # The name of our trained model
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message=commit_message)






