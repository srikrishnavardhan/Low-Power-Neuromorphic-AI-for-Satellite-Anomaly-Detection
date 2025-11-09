#!/usr/bin/env python
# coding: utf-8

# In[2]:


# model_utils.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import io

# Device selection: use CUDA when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Define Models
# -------------------------------
class ANN_Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ANN_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class SNN_Model(nn.Module):
    def __init__(self, input_dim, num_classes, num_steps=10):
        super(SNN_Model, self).__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, 64)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan(), init_hidden=False)
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan(), init_hidden=False)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        mem1 = torch.zeros(batch_size, 64, device=x.device)
        mem2 = torch.zeros(batch_size, 32, device=x.device)
        mem3 = torch.zeros(batch_size, self.fc3.out_features, device=x.device)
        spk_sum = torch.zeros_like(mem3)

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            mem3 = mem3 * 0.9 + cur3
            spk_sum += mem3
        return spk_sum / self.num_steps

# -------------------------------
# Core Pipeline
# -------------------------------
def run_experiment(file, label_col="anomaly", epochs=50, batch_size=16):
    df = pd.read_csv(file)
    drop_cols = ['segment', 'train', 'channel', 'sampling']
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    num_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Keep tensors on CPU for DataLoader (batches will be moved to device during training)
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_test_t  = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.long)

    # Use pinned memory for faster host->GPU transfers when CUDA is available
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    results = []

    # ANN
    model_ann = ANN_Model(X_train.shape[1], num_classes).to(device)
    optimizer = optim.Adam(model_ann.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model_ann.train()
        for xb, yb in train_loader:
            # ensure batch is on correct device (DataLoader returns tensors as provided)
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model_ann(xb), yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = torch.argmax(model_ann(X_test_t.to(device)), dim=1).cpu()
        # y_test is a pandas series (cpu) so cast preds to cpu before metric calculation
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
    results.append(["ANN", acc, prec, rec, f1, (X_train.shape[1]*64 + 64*32 + 32*num_classes)*epochs])

    # SNN
    model_snn = SNN_Model(X_train.shape[1], num_classes).to(device)
    optimizer = optim.Adam(model_snn.parameters(), lr=5e-4)
    criterion_snn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model_snn.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion_snn(model_snn(xb), yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds_snn = torch.argmax(model_snn(X_test_t.to(device)), dim=1).cpu()
        acc_snn = accuracy_score(y_test, preds_snn)
        f1_snn = f1_score(y_test, preds_snn, average='weighted')
        prec_snn = precision_score(y_test, preds_snn, average='weighted')
        rec_snn = recall_score(y_test, preds_snn, average='weighted')
    spike_count = torch.abs(model_snn(X_test_t.to(device))).sum().item()
    energy_snn = spike_count / len(X_test)

    results.append(["SNN", acc_snn, prec_snn, rec_snn, f1_snn, energy_snn])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Energy"])
    return results_df

