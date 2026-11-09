from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)



ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Transactions.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 100_000
EPOCHS = 20
HIDDEN_CHANNELS = 32
LEARNING_RATE = 0.005
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cpu")




print(f"Loading at most {MAX_ROWS:,} transactions...")

df = pd.read_csv(DATA_PATH, nrows=MAX_ROWS)

required_columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "oldbalanceDest",
    "nameDest",
    "isFraud",
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.sort_values("step").reset_index(drop=True)

unique_steps = np.sort(df["step"].unique())

if len(unique_steps) < 2:
    raise ValueError("Not enough distinct steps for a chronological split.")

cutoff_index = int(len(unique_steps) * 0.8)
cutoff_index = min(max(cutoff_index, 1), len(unique_steps) - 1)
cutoff_step = unique_steps[cutoff_index]

train_mask = (df["step"] < cutoff_step).to_numpy()
test_mask = ~train_mask

train_positions = np.flatnonzero(train_mask)
test_positions = np.flatnonzero(test_mask)

y_all = df["isFraud"].to_numpy(dtype=np.float32)
y_train = y_all[train_positions]
y_test = y_all[test_positions]

print(f"Transactions loaded: {len(df):,}")
print(f"Training transactions: {len(train_positions):,}")
print(f"Testing transactions: {len(test_positions):,}")
print(f"Chronological cutoff step: {cutoff_step}")
print(f"Training fraud cases: {int(y_train.sum()):,}")
print(f"Testing fraud cases: {int(y_test.sum()):,}")

if len(np.unique(y_train)) < 2:
    raise ValueError(
        "Training data contains only one class. Increase MAX_ROWS."
    )

if y_test.sum() == 0 or y_test.sum() == len(y_test):
    raise ValueError(
        "The test set needs both fraud and legitimate transactions. "
        "Increase MAX_ROWS to include more simulation steps."
    )


accounts = pd.Index(
    pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True).unique()
)

account_to_id = {account: i for i, account in enumerate(accounts)}

source_ids = df["nameOrig"].map(account_to_id).to_numpy(dtype=np.int64)
destination_ids = df["nameDest"].map(account_to_id).to_numpy(dtype=np.int64)

num_nodes = len(accounts)

print(f"Unique account nodes: {num_nodes:,}")


train_sources = source_ids[train_positions]
train_destinations = destination_ids[train_positions]

edge_index = torch.tensor(
    np.vstack(
        [
            np.concatenate([train_sources, train_destinations]),
            np.concatenate([train_destinations, train_sources]),
        ]
    ),
    dtype=torch.long,
    device=DEVICE,
)


node_features = np.zeros((num_nodes, 4), dtype=np.float32)

train_amounts = df["amount"].to_numpy(dtype=np.float64)[train_positions]

out_count = np.bincount(train_sources, minlength=num_nodes)
in_count = np.bincount(train_destinations, minlength=num_nodes)

out_amount = np.bincount(
    train_sources,
    weights=train_amounts,
    minlength=num_nodes,
)
in_amount = np.bincount(
    train_destinations,
    weights=train_amounts,
    minlength=num_nodes,
)

node_features[:, 0] = np.log1p(out_count)
node_features[:, 1] = np.log1p(in_count)
node_features[:, 2] = np.log1p(out_amount)
node_features[:, 3] = np.log1p(in_amount)


feature_mean = node_features.mean(axis=0)
feature_std = node_features.std(axis=0)
feature_std[feature_std == 0] = 1.0

node_features = (node_features - feature_mean) / feature_std

x = torch.tensor(node_features, dtype=torch.float32, device=DEVICE)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)


amount = df["amount"].to_numpy(dtype=np.float32)
origin_balance = df["oldbalanceOrg"].to_numpy(dtype=np.float32)
destination_balance = df["oldbalanceDest"].to_numpy(dtype=np.float32)

transaction_features = pd.DataFrame(
    {
        "log_amount": np.log1p(np.maximum(amount, 0)),
        "log_origin_balance": np.log1p(np.maximum(origin_balance, 0)),
        "log_destination_balance": np.log1p(
            np.maximum(destination_balance, 0)
        ),
        "amount_to_origin_balance": amount
        / np.maximum(origin_balance, 1.0),
    }
)

transaction_features["transaction_type"] = df["type"].astype(str)

transaction_features = pd.get_dummies(
    transaction_features,
    columns=["transaction_type"],
    dtype=np.float32,
)

transaction_x = transaction_features.to_numpy(dtype=np.float32)


tx_mean = transaction_x[train_positions].mean(axis=0)
tx_std = transaction_x[train_positions].std(axis=0)
tx_std[tx_std == 0] = 1.0

transaction_x = (transaction_x - tx_mean) / tx_std

transaction_x = torch.tensor(
    transaction_x, dtype=torch.float32, device=DEVICE
)

all_sources = torch.tensor(source_ids, dtype=torch.long, device=DEVICE)
all_destinations = torch.tensor(
    destination_ids, dtype=torch.long, device=DEVICE
)

train_idx = torch.tensor(train_positions, dtype=torch.long, device=DEVICE)
test_idx = torch.tensor(test_positions, dtype=torch.long, device=DEVICE)


class GraphSAGEFraudModel(nn.Module):
    def __init__(self, node_input, hidden, transaction_input):
        super().__init__()

        self.conv1 = SAGEConv(node_input, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2 + transaction_input, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def encode(self, node_x, graph_edges):
        h = self.conv1(node_x, graph_edges)
        h = torch.relu(h)
        h = self.conv2(h, graph_edges)
        return torch.relu(h)

    def forward(
        self,
        node_x,
        graph_edges,
        sources,
        destinations,
        tx_features,
    ):
        embeddings = self.encode(node_x, graph_edges)

        edge_features = torch.cat(
            [
                embeddings[sources],
                embeddings[destinations],
                tx_features,
            ],
            dim=1,
        )

        return self.classifier(edge_features).squeeze(1)


model = GraphSAGEFraudModel(
    node_input=x.shape[1],
    hidden=HIDDEN_CHANNELS,
    transaction_input=transaction_x.shape[1],
).to(DEVICE)

num_fraud = float(y_train.sum())
num_legitimate = float(len(y_train) - y_train.sum())

pos_weight = torch.tensor(
    [num_legitimate / max(num_fraud, 1.0)],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("\nTraining GraphSAGE...")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model(
        x,
        edge_index,
        all_sources[train_idx],
        all_destinations[train_idx],
        transaction_x[train_idx],
    )

    loss = criterion(logits, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {loss.item():.4f}"
        )


model.eval()

with torch.no_grad():
    test_logits = model(
        x,
        edge_index,
        all_sources[test_idx],
        all_destinations[test_idx],
        transaction_x[test_idx],
    )

    test_probabilities = torch.sigmoid(test_logits).cpu().numpy()



from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


print("\n--- GraphSAGE Threshold Comparison ---")
print(f"PR-AUC: {average_precision_score(y_test, test_probabilities):.4f}")

thresholds = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
selected_threshold = 0.7

for threshold in thresholds:
    predictions = (test_probabilities >= threshold).astype(np.int32)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"Threshold={threshold:.3f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f} | "
        f"FP={fp} | FN={fn} | TP={tp}"
    )

y_pred = (test_probabilities >= selected_threshold).astype(np.int32)

print(f"\nConfusion matrix at threshold {selected_threshold}:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))


artifact = {
    "model_state_dict": model.state_dict(),
    "node_feature_mean": feature_mean,
    "node_feature_std": feature_std,
    "transaction_feature_mean": tx_mean,
    "transaction_feature_std": tx_std,
    "transaction_feature_columns": transaction_features.columns.tolist(),
    "hidden_channels": HIDDEN_CHANNELS,
    "input_node_features": x.shape[1],
    "input_transaction_features": transaction_x.shape[1],
    "cutoff_step": int(cutoff_step),
    "max_rows": MAX_ROWS,
}

model_path = MODEL_DIR / "graphsage_fraud_model.pt"
torch.save(artifact, model_path)

print(f"\nModel saved to: {model_path}")