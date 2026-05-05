from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT_DIR / "snapshots"
OUTPUT_PATH = SNAPSHOT_DIR / "deep_learning_test_results.json"


def build_synthetic_tyre_data(rows: int = 420) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    tread_depth_mm = rng.uniform(4.0, 9.0, rows)
    load_index = rng.uniform(82, 104, rows)
    speed_rating = rng.integers(1, 6, rows)
    wet_grip_score = rng.uniform(1, 5, rows)
    rolling_resistance = rng.uniform(1, 5, rows)
    noise_db = rng.uniform(67, 75, rows)
    price_eur = rng.uniform(55, 220, rows)

    features = np.column_stack(
        [
            tread_depth_mm,
            load_index,
            speed_rating,
            wet_grip_score,
            rolling_resistance,
            noise_db,
            price_eur,
        ]
    ).astype("float32")

    useful_life_km = (
        9_500
        + tread_depth_mm * 4_250
        + load_index * 115
        + wet_grip_score * 520
        - rolling_resistance * 640
        - noise_db * 55
        + np.sqrt(price_eur) * 820
        + rng.normal(0, 1_100, rows)
    ).astype("float32")

    features = (features - features.mean(axis=0)) / features.std(axis=0)
    target = ((useful_life_km - useful_life_km.mean()) / useful_life_km.std()).reshape(-1, 1)
    return features, target.astype("float32")


def train_pytorch(features: np.ndarray, target: np.ndarray) -> dict:
    import torch

    torch.manual_seed(42)
    x = torch.tensor(features)
    y = torch.tensor(target)
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = torch.nn.MSELoss()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    final_loss = float(loss_fn(model(x), y).detach().cpu().item())
    return {
        "framework": "pytorch",
        "version": torch.__version__,
        "final_mse": round(final_loss, 6),
        "status": "passed" if final_loss < 0.16 else "warning",
    }


def train_tensorflow(features: np.ndarray, target: np.ndarray) -> dict:
    import tensorflow as tf

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(features.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03), loss="mse")
    history = model.fit(features, target, epochs=80, verbose=0)
    final_loss = float(history.history["loss"][-1])
    return {
        "framework": "tensorflow",
        "version": tf.__version__,
        "final_mse": round(final_loss, 6),
        "status": "passed" if final_loss < 0.16 else "warning",
    }


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    features, target = build_synthetic_tyre_data()
    results = {
        "dataset": {
            "rows": int(features.shape[0]),
            "features": int(features.shape[1]),
            "target": "standardized useful_life_km",
        },
        "tests": [
            train_pytorch(features, target),
            train_tensorflow(features, target),
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
