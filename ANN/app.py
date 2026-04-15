import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(APP_DIR, "templates"))


@dataclass
class TrainConfig:
    # Model
    hidden_layers: int
    hidden_units: int
    activation: str
    batch_norm: bool
    dropout: float
    weight_init: str

    # Training
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    l2_weight_decay: float
    momentum: float
    nesterov: bool
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism helps comparisons between runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unknown activation: {name}")


def init_linear_weights(layer: nn.Module, method: str) -> None:
    if not isinstance(layer, nn.Linear):
        return

    method = method.lower()
    if method == "xavier_uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif method == "xavier_normal":
        nn.init.xavier_normal_(layer.weight)
    elif method == "kaiming_uniform":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    elif method == "kaiming_normal":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    elif method == "normal":
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    elif method == "uniform":
        nn.init.uniform_(layer.weight, a=-0.05, b=0.05)
    elif method == "zeros":
        nn.init.zeros_(layer.weight)
    else:
        raise ValueError(f"Unknown weight_init: {method}")

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: int,
        hidden_units: int,
        num_classes: int,
        activation: str,
        batch_norm: bool,
        dropout: float,
        weight_init: str,
    ):
        super().__init__()
        activation_module_factory = lambda: get_activation(activation)

        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(activation_module_factory())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_units

        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

        # Initialize weights once (fresh run per training request).
        self.apply(lambda m: init_linear_weights(m, weight_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_bool(x: object) -> bool:
    # HTML checkboxes come through as "on"/"" (or absent).
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "on"}


def build_config_from_request() -> TrainConfig:
    form = request.form
    # Keep defaults aligned with the frontend defaults.
    return TrainConfig(
        hidden_layers=int(form.get("hidden_layers", 2)),
        hidden_units=int(form.get("hidden_units", 64)),
        activation=str(form.get("activation", "relu")),
        batch_norm=parse_bool(form.get("batch_norm", False)),
        dropout=float(form.get("dropout", 0.0)),
        weight_init=str(form.get("weight_init", "xavier_uniform")),
        optimizer=str(form.get("optimizer", "adam")),
        learning_rate=float(form.get("learning_rate", 0.001)),
        batch_size=int(form.get("batch_size", 64)),
        epochs=int(form.get("epochs", 20)),
        l2_weight_decay=float(form.get("l2_weight_decay", 0.0)),
        momentum=float(form.get("momentum", 0.9)),
        nesterov=parse_bool(form.get("nesterov", False)),
        seed=int(form.get("seed", 42)),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/train")
def train():
    cfg = build_config_from_request()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One dataset for all runs (as requested).
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )

    # Standardize features using train statistics only.
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_layers=cfg.hidden_layers,
        hidden_units=cfg.hidden_units,
        num_classes=int(np.max(y) + 1),
        activation=cfg.activation,
        batch_norm=cfg.batch_norm,
        dropout=cfg.dropout,
        weight_init=cfg.weight_init,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.l2_weight_decay,
        )
    elif cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.l2_weight_decay,
        )
    elif cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.l2_weight_decay,
        )
    elif cfg.optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.l2_weight_decay,
        )
    else:
        return jsonify({"ok": False, "error": f"Unknown optimizer: {cfg.optimizer}"}), 400

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            total += int(xb.size(0))

        train_loss = running_loss / max(total, 1)
        _, train_acc = evaluate(model, train_loader, device)
        _, test_acc = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )

    # Return final epoch metrics for the UI.
    final = history[-1] if history else {}
    payload = {
        "ok": True,
        "device": str(device),
        "config": cfg.__dict__,
        "final": {
            "epoch": final.get("epoch"),
            "train_loss": final.get("train_loss"),
            "train_acc": final.get("train_acc"),
            "test_acc": final.get("test_acc"),
        },
        "history": history[-10:],  # keep response small
    }
    return app.response_class(
        response=json.dumps(payload),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    # Usage:
    #   cd ANN
    #   python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)

