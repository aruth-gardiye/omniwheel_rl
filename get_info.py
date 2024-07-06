# Description: 
#   Code to get the average time and reward received per episode during training 
#   and average time and steps taken to reach target during testing

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Set the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Function display average time taken for training and average reward received per episode
def get_time(path):
    csv_file = list(Path(path).rglob("*.csv"))[-1]
    
    df = pd.read_csv(csv_file)
    
    method = df['method'][0]
    
    # Get the total time taken for training
    time = df['time_taken'].sum()
    
    task = df['task'][0]

    print(f'\nTraining info for {method} {task} (for all episodes)')
    
    print(f'Time taken for training: {time} seconds')
    
    # Get the average reward received per episode
    print(f'Average reward received per episode: {df["total_reward"].mean()}')
    
    # Get the standard deviation of reward received per episode
    print(f'Standard deviation of reward received per episode: {df["total_reward"].std()}\n')


# Function to display average time taken to reach target and average number of steps taken to reach target per episode
def get_time_steps(path):
    
    csv_files = list(Path(path).rglob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory {path}")
    
    df = pd.concat(
        [
            pd.read_csv(p)[pd.read_csv(p)["result"] == "success"]
            for p in csv_files
        ],
        ignore_index=True,
    )
    
    method = df['method'][0]
    
    task = df['task'][0]
    
    # Get the average time taken to reach target
    time = df['time_taken'].mean()

    # Get average reward received per episode
    reward = df['total_reward'].mean()

    # Get standard deviation of reward received per episode
    std = df['total_reward'].std()
    
    # Get the average number of steps taken to reach target
    steps = df['steps'].mean()

    print(f'\nTesting info for {method} {task} (successful episodes only)')
    
    print(f'Average time taken to reach target: {time} seconds')
    
    print(f'Average number of steps taken to reach target: {steps}')

    print(f'Average reward received per episode: {reward}')

    print(f'Standard deviation of reward received per episode: {std}\n')

# Get the average time taken for training and average reward received per episode during training
get_time(os.path.join(current_dir, "results/qlearning_static/train/"))
get_time(os.path.join(current_dir, "results/qlearning_moving/train/"))
get_time(os.path.join(current_dir, "results/actor_critic_moving/train/"))

# Get the average time taken to reach target and average number of steps taken to reach target per episode during testing
get_time_steps(os.path.join(current_dir, "results/qlearning_static/test/"))
get_time_steps(os.path.join(current_dir, "results/qlearning_moving/test/"))
get_time_steps(os.path.join(current_dir, "results/actor_critic_moving/test/"))