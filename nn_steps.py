# Description: Implementation of neural network to predict the number of steps to reach the target

import numpy as np
import pandas as pd
import os
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn, optim
import matplotlib.pyplot as plt

# Set the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Function to read data from CSV files and create a DataFrame with only successful episodes
def createDf(path):
    csv_files = list(Path(path).rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory {path}")
    return pd.concat(
        [
            pd.read_csv(p)[pd.read_csv(p)["result"] == "success"]
            for p in csv_files
        ],
        ignore_index=True,
    )

# Create dataframes
df_qls = createDf(
    os.path.join(current_dir, "results/qlearning_static/test/")
)
df_qlm = createDf(
    os.path.join(current_dir, "results/qlearning_moving/test/")
)
df_acm = createDf(
    os.path.join(current_dir, "results/actor_critic_moving/test/")
)

# Check if any of the dataframes are empty
if df_qls.empty or df_qlm.empty or df_acm.empty:
    raise ValueError("One or more dataframes are empty due to no successful episodes")

# Combine all dataframes into one and drop unnecessary columns
df = pd.concat([df_qls, df_qlm, df_acm], ignore_index=True).drop(
    columns=["episode", "time_taken", "total_reward", "result"]
)

# Define the ratio for train, dev, and test
ratios = [0.7, 0.15, 0.15]


# Function to split the data into train, dev, and test
def split_data(df, ratios):
    train, validate, test = np.split(
        df.sample(frac=1),
        [int(ratios[0] * len(df)), int((ratios[0] + ratios[1]) * len(df))],
    )
    return train, validate, test


# Split the data
train_df, dev_df, test_df = split_data(df, ratios)

# check how many rows are in each dataframe grouped by task and method
print(f'\nTrain:\n {train_df.groupby(["task", "method"]).size()}')
print(f'\nDev:\n {dev_df.groupby(["task", "method"]).size()}')
print(f'\nTest:\n {test_df.groupby(["task", "method"]).size()}\n')


# Function to preprocess the data
def preprocess_data(df):
    # Convert to float
    df[
        [
            "initial_robot_position_x",
            "initial_robot_position_y",
            "initial_target_poisition_x",
            "initial_target_poisition_y",
            "steps",
        ]
    ] = df[
        [
            "initial_robot_position_x",
            "initial_robot_position_y",
            "initial_target_poisition_x",
            "initial_target_poisition_y",
            "steps",
        ]
    ].astype(
        float
    )

    # Encode task and method
    le = LabelEncoder()
    df["task"] = le.fit_transform(df["task"])
    df["method"] = le.fit_transform(df["method"])

    # Scale positions
    scaler = MinMaxScaler()
    df[
        [
            "initial_robot_position_x",
            "initial_robot_position_y",
            "initial_target_poisition_x",
            "initial_target_poisition_y",
        ]
    ] = scaler.fit_transform(
        df[
            [
                "initial_robot_position_x",
                "initial_robot_position_y",
                "initial_target_poisition_x",
                "initial_target_poisition_y",
            ]
        ]
    )

    return df


# Preprocess the data
train_df = preprocess_data(train_df)
dev_df = preprocess_data(dev_df)
test_df = preprocess_data(test_df)

# Convert to PyTorch tensors
X_train = torch.tensor(train_df.drop(columns=["steps"]).values, dtype=torch.float32)
y_train = torch.tensor(train_df["steps"].values, dtype=torch.float32).reshape(-1, 1)
X_dev = torch.tensor(dev_df.drop(columns=["steps"]).values, dtype=torch.float32)
y_dev = torch.tensor(dev_df["steps"].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(test_df.drop(columns=["steps"]).values, dtype=torch.float32)
y_test = torch.tensor(test_df["steps"].values, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(6, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1),
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 1000  # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    for start in batch_start:
        # take a batch
        X_batch = X_train[start : start + batch_size]
        y_batch = y_train[start : start + batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_dev_pred = model(X_dev)
        dev_loss = loss_fn(y_dev_pred, y_dev)
        if dev_loss < best_mse:
            best_mse = dev_loss
            best_weights = model.state_dict().copy()

    history.append((loss.item(), dev_loss.item()))
    print(
        f"Epoch {epoch+1}/{n_epochs}, Training Loss: {loss.item()}, Validation Loss: {dev_loss.item()}"
    )

model.load_state_dict(best_weights)

# Test and get RMSE for test set
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = loss_fn(y_test_pred, y_test)
    test_rmse = torch.sqrt(test_loss)
    print(f"Test RMSE: {test_rmse:.2f}")


# Create plot of training and validation loss
train_losses, val_losses = zip(*history)

plt.figure()

# Plot training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')

plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

# Create folder to save plots
directory = os.path.join(current_dir, 'results/nn_predict_steps')
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the plot
plt.savefig(f'{directory}/loss.png')


# Create a plot to compare the predicted and actual steps
plt.figure()

# Plot predicted and actual steps
plt.plot(y_test.detach().numpy(), label='Actual Steps')
plt.plot(y_test_pred.detach().numpy(), label='Predicted Steps')

plt.title('Predicted and Actual Steps')
plt.xlabel('Episodes')
plt.ylabel('Steps')

plt.legend()

# Save the plot
plt.savefig(f'{directory}/predicted_steps.png')