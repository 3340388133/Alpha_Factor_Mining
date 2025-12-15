import numpy as np
from typing import Tuple, List


def create_synthetic_data(T: int = 500, N: int = 100, F: int = 10, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    np.random.seed(seed)
    feature_names = ["close", "open", "high", "low", "volume", "vwap", "returns", "volatility", "momentum", "turnover"][:F]
    while len(feature_names) < F:
        feature_names.append(f"feature_{len(feature_names)}")

    data = np.random.randn(T, N, F)
    trend = np.cumsum(np.random.randn(T, N) * 0.01, axis=0)
    data[:, :, 0] = 100 + trend + np.random.randn(T, N) * 0.5

    if F > 1: data[1:, :, 1] = data[:-1, :, 0] + np.random.randn(T-1, N) * 0.3; data[0, :, 1] = data[0, :, 0]
    if F > 2: data[:, :, 2] = np.maximum(data[:, :, 0], data[:, :, 1]) + np.abs(np.random.randn(T, N)) * 0.5
    if F > 3: data[:, :, 3] = np.minimum(data[:, :, 0], data[:, :, 1]) - np.abs(np.random.randn(T, N)) * 0.5
    if F > 4: data[:, :, 4] = np.maximum(np.cumsum(np.random.randn(T, N) * 50000, axis=0) + 1e6, 1e5)
    if F > 5: data[:, :, 5] = data[:, :, 0] * 0.4 + data[:, :, 2] * 0.3 + data[:, :, 3] * 0.3
    if F > 6:
        data[1:, :, 6] = (data[1:, :, 0] - data[:-1, :, 0]) / data[:-1, :, 0]
        data[0, :, 6] = 0
    if F > 7:
        for t in range(20, T): data[t, :, 7] = np.std(data[t-20:t, :, 6], axis=0)
        data[:20, :, 7] = data[20, :, 7]
    if F > 8:
        for t in range(10, T): data[t, :, 8] = (data[t, :, 0] - data[t-10, :, 0]) / data[t-10, :, 0]
    if F > 9:
        shares = np.random.uniform(1e8, 1e9, N)
        data[:, :, 9] = data[:, :, 4] / shares

    returns = data[:, :, 6].copy() if F > 6 else np.diff(data[:, :, 0], axis=0, prepend=data[0:1, :, 0]) / np.maximum(data[:, :, 0], 1)
    return data, returns, feature_names


def split_data(data: np.ndarray, returns: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    T = data.shape[0]
    t1, t2 = int(T * train_ratio), int(T * (train_ratio + val_ratio))
    return {"train": {"data": data[:t1], "returns": returns[:t1]},
            "val": {"data": data[t1:t2], "returns": returns[t1:t2]},
            "test": {"data": data[t2:], "returns": returns[t2:]}}
