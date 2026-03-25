"""Tiny Transformer for sequence-based direction classification on CUDA."""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class TinyTransformer(nn.Module):
    """Small transformer encoder for time-series classification."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        x = self.proj(x)
        x = self.pos(x)
        x = self.enc(x)
        x = x.mean(dim=1)  # mean pool over time
        return self.head(x)


class TransformerDirectionModel:
    """Wrapper around TinyTransformer for training and inference."""

    def __init__(
        self,
        d_in: int,
        seq_len: int = 60,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 20,
        batch_size: int = 256,
    ):
        self.d_in = d_in
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: TinyTransformer | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> TinyTransformer:
        model = TinyTransformer(
            d_in=self.d_in,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
        )
        return model.to(self.device)

    def train(self, X_sequences: np.ndarray, y: np.ndarray) -> dict:
        """Train the transformer.

        Args:
            X_sequences: (N, seq_len, n_features).
            y: Labels in {-1, 0, 1}.

        Returns:
            Dict with per-epoch train_loss, val_loss, val_accuracy.
        """
        y_mapped = y + 1  # {-1,0,1} -> {0,1,2}

        # Compute class weights (inverse frequency)
        classes, counts = np.unique(y_mapped, return_counts=True)
        total = len(y_mapped)
        weight_arr = np.ones(3, dtype=np.float32)
        for cls, cnt in zip(classes, counts):
            weight_arr[int(cls)] = total / (3.0 * cnt)
        class_weights = torch.tensor(weight_arr, dtype=torch.float32).to(self.device)

        # Split: last 20% as validation
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_val = y_mapped[:split_idx], y_mapped[split_idx:]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.model = self._build_model()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_train_loss = epoch_loss / max(n_batches, 1)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    val_loss += criterion(logits, yb).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_acc = correct / max(total, 1)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_acc)

            logger.info("Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_acc=%.4f",
                         epoch + 1, self.epochs, avg_train_loss, avg_val_loss, val_acc)

        return history

    def predict_proba(self, X_sequences: np.ndarray) -> np.ndarray:
        """Return (N, 3) softmax probabilities [p_down, p_flat, p_up]."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        self.model.eval()
        X_t = torch.tensor(X_sequences, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probas = torch.softmax(logits, dim=1).cpu().numpy()
        return probas

    def predict(self, X_sequences: np.ndarray) -> np.ndarray:
        """Return class predictions in {-1, 0, 1}."""
        probas = self.predict_proba(X_sequences)
        return probas.argmax(axis=1).astype(int) - 1  # {0,1,2} -> {-1,0,1}

    def save(self, path: str) -> None:
        """Save model state dict and hyperparams."""
        if self.model is None:
            raise RuntimeError("Model not trained — nothing to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hyperparams": {
                    "d_in": self.d_in,
                    "seq_len": self.seq_len,
                    "d_model": self.d_model,
                    "n_heads": self.n_heads,
                    "n_layers": self.n_layers,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )
        logger.info("Transformer model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model state dict and rebuild architecture."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        hp = checkpoint["hyperparams"]
        self.d_in = hp["d_in"]
        self.seq_len = hp["seq_len"]
        self.d_model = hp["d_model"]
        self.n_heads = hp["n_heads"]
        self.n_layers = hp["n_layers"]
        self.lr = hp["lr"]
        self.weight_decay = hp["weight_decay"]
        self.epochs = hp["epochs"]
        self.batch_size = hp["batch_size"]
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        logger.info("Transformer model loaded from %s", path)
