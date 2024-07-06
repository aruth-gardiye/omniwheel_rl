# Description: Implementation of Actor-Critic algorithm to solve the moving target task.

import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

from env import Environment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Define the constants for the environment and agent
state_size = (
    7  # (robot_x, robot_y, robot_theta, target_x, target_y, target_vel_x, target_vel_y)
)
action_size = 9
learning_rate = 1e-2
discount_factor = 0.99
num_episodes = 1000
num_test_episodes = 50
max_steps = 1000
current_dir = os.path.dirname(os.path.realpath(__file__))

# initialize pygame
pygame.init()


# Define the Agent class
class ActorCriticAgent:
    def __init__(
        self, state_size, action_size, learning_rate=1e-2, gamma=0.99, lr_decay=0.99
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lr_decay = lr_decay

        # lists for saving actions and rewards for an episode
        self.saved_actions = []
        self.rewards = []

        # Define the actor and critic networks
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=0),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Define the optimizer for the actor and critic networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # smallest useful value
        self.eps = np.finfo(np.float32).eps.item()

        # Define the actions
        self.actions = np.array(
            [
                # up
                [0, 0.5, 0],
                # up right
                [0.5, 0, -0.5],
                # right
                [0.5, -0.5, 0],
                # down right
                [0, -0.5, 0.5],
                # down
                [0, -0.5, 0],
                # down left
                [-0.5, 0, 0.5],
                # left
                [-0.5, 0.5, 0],
                # up left
                [0, 0.5, -0.5],
                # rotate in place
                [0.5, 0.5, 0.5],
            ]
        )

    def choose_action(self, state):
        # state = robot_x, robot_y, robot_theta, target_x, target_y, target_vel_x, target_vel_y
        state = torch.from_numpy(state).float().unsqueeze(0)

        # Get the action probabilities and state value from the actor and critic networks
        action_probs = self.actor(state)
        state_value = self.critic(state)

        # Choose a discrete action based on the action probabilities
        m = Categorical(action_probs)
        action_index = m.sample().item()

        # Map the index to the continuous action
        action = self.actions[action_index]
        action_index = torch.tensor([action_index])

        # Save the action probability and state value for training
        self.saved_actions.append(SavedAction(m.log_prob(action_index), state_value))

        # choose action
        return action

    def update(self, state, action, reward, next_state, done):
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        state = torch.from_numpy(state).float().unsqueeze(0)

        # Calculate the state value of the next state
        next_state_value = self.critic(next_state)

        # Calculate the state value of the current state
        state_value = self.critic(state)

        # Calculate the TD error
        td_error = reward + self.gamma * (next_state_value * (1 - done)) - state_value

        # Calculate the loss of the actor network
        actor_loss = -self.saved_actions[-1].log_prob * td_error.detach()

        # Calculate the loss of the critic network
        critic_loss = td_error**2

        # Update the learning rate of the optimizer for both the actor and critic networks
        self.actor_optimizer.param_groups[0]["lr"] *= self.lr_decay
        self.critic_optimizer.param_groups[0]["lr"] *= self.lr_decay

        # Update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Clear the saved actions
        del self.saved_actions[:]
        del self.rewards[:]


# Define the class for the saved actions
class SavedAction:
    def __init__(self, log_prob, state_value):
        self.log_prob = log_prob
        self.state_value = state_value


# Initialize the environment and agent
env = Environment()
agent = ActorCriticAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=learning_rate,
    gamma=discount_factor,
)


def train_agent(agent, env, num_episodes, max_steps, save=True, render=False):
    # Define the DataFrame to store the train results
    train_total_rewards_steps = pd.DataFrame(
        columns=[
            "task",
            "method",
            "episode",
            "time_taken",
            "initial_robot_position_x",
            "initial_robot_position_y",
            "initial_target_poisition_x",
            "initial_target_poisition_y",
            "total_reward",
            "steps",
            "result",
        ]
    )

    # Reset the environment for a new episode
    env.reset(first_episode=True)

    # Train the agent using Q-learning
    for episode in range(num_episodes):
        # Record the start time of the episode
        start_time = time.time()

        # Reset the environment for a new episode
        state = env.reset()

        # Get the initial robot and target positions
        state = env.get_observation()

        # Set the total reward to 0
        total_reward = 0

        # Run the episode until the agent reaches the goal or a maximum number of steps is reached
        while True:
            # Choose an action based on the current state
            action = agent.choose_action(state)

            omega_0, omega_1, omega_2 = action

            # Take the action and observe the next state and reward
            next_state, reward, done = env.step(omega_0, omega_1, omega_2, render)

            # Save the next state value
            next_state_value = next_state

            # Get the observation based on the next state
            next_state = env.get_observation()

            # Update total reward
            total_reward += reward

            # Check if the agent has reached the goal or a maximum number of steps is reached
            if done or env.time_step == max_steps - 1:
                # Calculate the time it took to train the episode
                end_time = time.time()
                time_taken = end_time - start_time

                # Append the results to the DataFrame
                if next_state_value == 0:
                    train_total_rewards_steps.loc[len(train_total_rewards_steps)] = [
                        "moving_target",
                        "actor_critic",
                        episode,
                        time_taken,
                        next_state[0],
                        next_state[1],
                        next_state[3],
                        next_state[4],
                        total_reward,
                        env.time_step,
                        "success",
                    ]
                else:
                    train_total_rewards_steps.loc[len(train_total_rewards_steps)] = [
                        "moving_target",
                        "actor_critic",
                        episode,
                        time_taken,
                        next_state[0],
                        next_state[1],
                        next_state[3],
                        next_state[4],
                        total_reward,
                        env.time_step,
                        "fail",
                    ]

                break

        # Update the agent's knowledge using the observed state, reward, and done values
        agent.update(state, action, reward, next_state, done)

        # Print the total reward after 10% of the episodes
        if episode % (num_episodes // 10) == 0:
            print(f"Episode {episode}: Total reward = {total_reward}")

    # Save the results to a CSV file if save is True, format csv file name with method and task and timestamp
    if save:
        # create folder for results "results/actor_critic_moving/"
        results_dir = os.path.join(current_dir, "results/actor_critic_moving/train/")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # save results to csv file
        train_total_rewards_steps.to_csv(
            f'{results_dir}actor_critic_moving_train_{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.csv',
            index=False,
            header=True,
        )

    return train_total_rewards_steps


def test_agent(agent, env, num_episodes, max_steps, save=True, render=False):
    # Define the DataFrame to store the test results
    test_total_rewards_steps = pd.DataFrame(
        columns=[
            "task",
            "method",
            "episode",
            "time_taken",
            "initial_robot_position_x",
            "initial_robot_position_y",
            "initial_target_poisition_x",
            "initial_target_poisition_y",
            "total_reward",
            "steps",
            "result",
        ]
    )

    # Set the actor and critic networks to evaluation mode
    agent.actor.eval()
    agent.critic.eval()

    # Reset the environment for a new episode
    env.reset(first_episode=True)

    for episode in range(num_episodes):
        # Record the start time of the episode
        start_time = time.time()

        # Reset the environment for a new episode
        state = env.reset()

        # Get initial state
        state = env.get_observation()

        # Get the initial robot and target positions
        positions = env.get_observation()

        # Set the total reward to 0
        total_reward = 0

        # Run the episode until the agent reaches the goal or a maximum number of steps is reached
        while True:
            # Choose an action based on the current state
            action = agent.choose_action(state)

            # Set wheel speeds
            omega_0, omega_1, omega_2 = action

            # Take the action and observe the next state and reward
            state, reward, done = env.step(omega_0, omega_1, omega_2, render)

            # Save the state value
            state_value = state

            # Get the observation based on the state
            state = env.get_observation()

            # Update total reward
            total_reward += reward

            # Check if the agent has reached the goal or a maximum number of steps is reached
            if done or env.time_step == max_steps - 1:
                # Calculate the time it took to train the episode
                end_time = time.time()
                time_taken = end_time - start_time

                print(f"Episode {episode}: Total reward = {total_reward}")

                # Append the results to the DataFrame
                if state_value == 0:
                    test_total_rewards_steps.loc[len(test_total_rewards_steps)] = [
                        "moving_target",
                        "actor_critic",
                        episode,
                        time_taken,
                        positions[0],
                        positions[1],
                        positions[3],
                        positions[4],
                        total_reward,
                        env.time_step,
                        "success",
                    ]
                else:
                    test_total_rewards_steps.loc[len(test_total_rewards_steps)] = [
                        "moving_target",
                        "actor_critic",
                        episode,
                        time_taken,
                        positions[0],
                        positions[1],
                        positions[3],
                        positions[4],
                        total_reward,
                        env.time_step,
                        "fail",
                    ]
                break

    # Calculate the Test-Average and Test-Standard-Deviation
    test_average = test_total_rewards_steps["total_reward"].mean()
    test_std_dev = test_total_rewards_steps["total_reward"].std()

    # Print the Test-Average and Test-Standard-Deviation
    print(f"Test-Average: {test_average}")
    print(f"Test-Standard-Deviation: {test_std_dev}")

    # Save the results to a CSV file if save is True, format csv file name with method and task and timestamp
    if save:
        # create folder for results "results/actor_critic_moving/"
        result_dir = os.path.join(current_dir, "results/actor_critic_moving/test/")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # save results to csv file
        test_total_rewards_steps.to_csv(
            f'{result_dir}actor_critic_moving_test_{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.csv',
            index=False,
            header=True,
        )

    return test_total_rewards_steps


def plot_results(results, type, show=False, save=False):
    # Set the directory to save the plots
    directory = os.path.join(current_dir, f"results/actor_critic_moving/plots/{type}/")

    # depend on the type (train or test) plot the results, get method and task
    method = results["method"][0]
    task = results["task"][0]

    # get the total reward and steps
    total_reward = results["total_reward"]
    steps = results["steps"]

    # plot the total reward
    plt.figure()
    plt.plot(results["episode"], total_reward)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Total Reward vs Episode for {type} ({method}) ({task})")
    if show:
        plt.show()
    if save:
        # create folder for plots
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save plot to png file to the folder
        plt.savefig(
            f'{directory}/Total Reward vs Episode for {type} ({method}) ({task})_{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.png',
            dpi=300,
        )

    # plot the steps
    plt.figure()
    plt.plot(results["episode"], steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title(f"Episode vs Steps  for {type} ({method}) ({task})")
    if show:
        plt.show()
    if save:
        # create folder for plots
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save plot to png file
        plt.savefig(
            f'{directory}/Episode vs Steps for {type} ({method}) ({task})_{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.png',
            dpi=300,
        )

    # plot the histogram of the total reward
    plt.figure()
    plt.hist(total_reward)
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title(f"Total Reward Histogram for {type} ({method}) ({task})")
    if show:
        plt.show()
    if save:
        # create folder for plots
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save plot to png file
        plt.savefig(
            f'{directory}/Total Reward Histogram for {type} ({method}) ({task})_{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.png',
            dpi=300,
        )


# Train the agent
train_results = train_agent(agent, env, num_episodes, max_steps)

# Plot the training results
plot_results(train_results, "train", save=True)

test_results_a = []

# Test the agent
test_results = test_agent(agent, env, num_test_episodes, max_steps, render=False)
test_results_a.append(test_results)

# Plot the test results
plot_results(test_results, "test", save=True)

# Test the agent again for 20 times
for i in range(20):
    test_results = test_agent(agent, env, 50, max_steps)
    test_results_a.append(test_results)

# count the number of success and fail
success = 0
fail = 0

for i in range(len(test_results_a)):
    success += len(test_results_a[i][test_results_a[i]["result"] == "success"])
    fail += len(test_results_a[i][test_results_a[i]["result"] == "fail"])

# print the number of success and fail
print(f"Success: {success}")
print(f"Fail: {fail}")
