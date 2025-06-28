# main.py
# --------------------------
# Refine-Recs - First ML model
# Just testing out a simple rec engine that learns what kinda movies a user might like
# --------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------
# Model: just a basic feedforward net
# Input = movie + user prefs combined
# Output = probability of "like"
# --------------------------
class MovieRecModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # squish output between 0 and 1
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# Data loader
# Assumes the processed CSV has only numeric data + 'liked' label
# --------------------------
def load_data():
    df = pd.read_csv("data/processed/movies_cleaned.csv")

    # Grab all columns except 'title' and the label
    feature_cols = [col for col in df.columns if col not in ['title', 'liked']]
    X = df[feature_cols].values.astype(np.float32)
    y = df['liked'].values.astype(np.float32).reshape(-1, 1)

    # Split for validation too
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Training function
# Classic PyTorch loop
# --------------------------
def train(model, X_train, y_train, X_val, y_val, epochs=20):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        # forward pass
        preds = model(torch.from_numpy(X_train))
        loss = loss_fn(preds, torch.from_numpy(y_train))

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.from_numpy(X_val))
            val_loss = loss_fn(val_preds, torch.from_numpy(y_val))
            acc = ((val_preds > 0.5).float() == torch.from_numpy(y_val)).float().mean()

        print(f"[{epoch+1}/{epochs}] Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {acc:.4f}")

# --------------------------
# Run it
# --------------------------
if __name__ == "__main__":
    # Load and split
    X_train, X_val, y_train, y_val = load_data()

    # Setup model
    input_size = X_train.shape[1]
    model = MovieRecModel(input_size)

    # Train
    train(model, X_train, y_train, X_val, y_val, epochs=25)
