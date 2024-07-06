# Description: Implementation of Q-learning algorithm for moving target task

import pygame
import numpy as np
from env import Environment
import pandas as pd
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Define the constants for the environment and agent
state_size = 5
action_size = 9
learning_rate = 0.1
discount_factor = 0.99
epsilion = 1.0
epsilion_decay_rate = 0.99
min_epsilion = 0.01
num_episodes = 1000
num_test_episodes = 50
max_steps = 1000
current_dir = os.path.dirname(os.path.realpath(__file__))

# initialize pygame
pygame.init()


# Define the Agent class
class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilion=1.0,
        epsilion_decay_rate=0.99,
        min_epsilion=0.01,
    ):
        # Define the size of the state and action spaces
        self.state_size = state_size
        self.action_size = action_size

        # Define the learning rate, discount factor, and exploration rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilion = epsilion
        self.epsilion_decay_rate = epsilion_decay_rate
        self.min_epsilion = min_epsilion

        # possible actions
        # ωi ∈ {−0.5, 0.0, 0.5} speed of each wheel
        # 9 possible actions to go in 8 directions and rotate in place
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
        self.current_action = self.actions[0]

        # Initialize the Q-table with possible states and actions)
        self.q_table = np.zeros((self.state_size, self.actions.shape[0]))

    def choose_action(self, state):
        try:
            # Choose an action based on the current state and the exploration rate
            if np.random.uniform(0, 1) < self.epsilion:
                # Choose a random action for the wheel speeds from self.actions
                action_index = np.random.choice(self.actions.shape[0])
                action = self.actions[action_index]

            else:
                # Choose the action with the highest Q-value
                action_index = np.argmax(self.q_table[state, :])
                action = self.actions[action_index]

            self.current_action = action
            return action

        except Exception as e:
            print(f"Error in choose_action: state={state}")
            raise e

    def update_q_table(self, state, action, reward, next_state):
        # match action to possible actions
        action_index = np.where((self.actions == action))

        # q(s,a) = q(s,a) + α(r + γ maxa' q(s',a') - q(s,a))
        self.q_table[state, action_index] = self.q_table[
            state, action_index
        ] + self.learning_rate * (
            reward
            + self.discount_factor * np.max(self.q_table[next_state, :])
            - self.q_table[state, action_index]
        )

    def decay_epsilion(self):
        # Decay the exploration rate over time
        self.epsilion *= self.epsilion_decay_rate
        self.epsilion = max(self.epsilion, self.min_epsilion)


# Initialize the environment and agent
env = Environment()
agent = Agent(
    state_size,
    action_size,
    learning_rate,
    discount_factor,
    epsilion,
    epsilion_decay_rate,
    min_epsilion,
)


# Define the function to train the agent
def train_agent(agent, env, num_episodes, max_steps, render=False, save=True):
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
        positions = env.get_observation()

        # Set the total reward to 0
        total_reward = 0

        # Run the episode until the agent reaches the goal or a maximum number of steps is reached
        while True:
            # Choose an action based on the current state
            action = agent.choose_action(state)

            # Take the action and observe the next state and reward
            next_state, reward, done = env.step(*action, render)

            # Update the Q-table based on the current state, action, reward, and next state
            agent.update_q_table(state, action, reward, next_state)

            # Decay the exploration rate over time
            agent.decay_epsilion()

            # Update total reward
            total_reward += reward

            # Check if the agent has reached the goal or a maximum number of steps is reached
            if done or env.time_step == max_steps - 1:
                # Calculate the time it took to train the episode
                end_time = time.time()
                time_taken = end_time - start_time

                # Append the results to the DataFrame
                if next_state == 0:
                    train_total_rewards_steps.loc[len(train_total_rewards_steps)] = [
                        "moving_target",
                        "qlearning",
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
                    train_total_rewards_steps.loc[len(train_total_rewards_steps)] = [
                        "moving_target",
                        "qlearning",
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

        # Print the total reward after 10% of the episodes
        if episode % (num_episodes // 10) == 0:
            print(f"Episode {env.episode}: Total reward = {total_reward}")

    # Save the results to a CSV file if save is True, format csv file name with method and task and timestamp
    if save:
        # create folder for results "results/qlearning_moving/train/"
        results_dir = os.path.join(current_dir, "results/qlearning_moving/train/")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # save results to csv file
        train_total_rewards_steps.to_csv(
            f'{results_dir}qlearning_moving_train_{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.csv',
            index=False,
            header=True,
        )

    return train_total_rewards_steps


def test_agent(agent, env, num_episodes, max_steps, render=False, save=True):
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

    # Set the exploration rate to 0
    agent.epsilion = 0

    # Reset the environment for a new episode
    env.reset(first_episode=True)

    for episode in range(num_episodes):
        # Record the start time of the episode
        start_time = time.time()

        # Reset the environment for a new episode
        state = env.reset()

        # Get the initial robot and target positions
        positions = env.get_observation()

        # Set the total reward to 0
        total_reward = 0

        # Run the episode until the agent reaches the goal or a maximum number of steps is reached
        while True:
            # Choose an action based on the current state
            action = agent.choose_action(state)

            # Take the action and observe the next state and reward
            state, reward, done = env.step(*action, render)

            # Update total reward
            total_reward += reward

            # Check if the agent has reached the goal or a maximum number of steps is reached
            if done or env.time_step == max_steps - 1:
                # Calculate the time it took to train the episode
                end_time = time.time()
                time_taken = end_time - start_time

                print(f"Episode {episode}: Total reward = {total_reward}")

                # Append the results to the DataFrame
                if state == 0:
                    test_total_rewards_steps.loc[len(test_total_rewards_steps)] = [
                        "moving_target",
                        "qlearning",
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
                        "qlearning",
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
        # create folder for results "results/qlearning_moving/test/"
        result_dir = os.path.join(current_dir, "results/qlearning_moving/test/")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # save results to csv file
        test_total_rewards_steps.to_csv(
            f'{result_dir}qlearning_moving_test_{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.csv',
            index=False,
            header=True,
        )

    return test_total_rewards_steps


def plot_results(results, type, show=False, save=False):
    # Set the directory to save the plots
    directory = os.path.join(current_dir, f"results/qlearning_moving/plots/{type}/")

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
