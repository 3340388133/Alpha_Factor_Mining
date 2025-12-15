import numpy as np
from typing import List

from .expression import OpType, ASTNode


class FactorCalculator:
    def __init__(self, data: np.ndarray, feature_names: List[str]):
        self.data = data  # shape (T, N, F)
        self.feature_names = feature_names
        self.name_to_idx = {n: i for i, n in enumerate(feature_names)}

    def _ts_op(self, x: np.ndarray, op: OpType, w: int) -> np.ndarray:
        T, N = x.shape
        w = max(1, int(w or 1))
        out = np.full_like(x, np.nan, dtype=float)
        for t in range(T):
            t0 = max(0, t - w + 1)
            window = x[t0:t + 1]
            if window.size == 0:
                continue
            if op == OpType.TS_MEAN:
                out[t] = np.nanmean(window, axis=0)
            elif op == OpType.TS_STD:
                out[t] = np.nanstd(window, axis=0)
            elif op == OpType.TS_MAX:
                out[t] = np.nanmax(window, axis=0)
            elif op == OpType.TS_MIN:
                out[t] = np.nanmin(window, axis=0)
            elif op == OpType.TS_RANK:
                last = window[-1]
                ranks = last.argsort().argsort().astype(float)
                out[t] = ranks / max(1, N - 1)
        return out

    def _cs_rank(self, x: np.ndarray) -> np.ndarray:
        T, N = x.shape
        out = np.full_like(x, np.nan, dtype=float)
        for t in range(T):
            row = x[t]
            valid = ~np.isnan(row)
            if valid.sum() == 0:
                continue
            ranks = row[valid].argsort().argsort().astype(float)
            out[t, valid] = ranks / max(1, valid.sum() - 1)
        return out

    def _cs_zscore(self, x: np.ndarray) -> np.ndarray:
        T, N = x.shape
        out = np.full_like(x, np.nan, dtype=float)
        for t in range(T):
            row = x[t]
            valid = ~np.isnan(row)
            if valid.sum() == 0:
                continue
            m = np.nanmean(row[valid])
            s = np.nanstd(row[valid])
            out[t, valid] = (row[valid] - m) / (s + 1e-8)
        return out

    def calculate(self, node: ASTNode) -> np.ndarray:
        op = node.op
        if op == OpType.FEATURE:
            idx = self.name_to_idx.get(str(node.value), None)
            if idx is None:
                return np.full(self.data.shape[:2], np.nan)
            return self.data[:, :, idx]
        if op == OpType.CONSTANT:
            return np.full(self.data.shape[:2], float(node.value))

        if op in {OpType.ABS, OpType.LOG, OpType.SIGN, OpType.CS_RANK, OpType.CS_ZSCORE,
                  OpType.TS_MEAN, OpType.TS_STD, OpType.TS_MAX, OpType.TS_MIN, OpType.TS_RANK}:
            x = self.calculate(node.children[0])
            if op == OpType.ABS:
                return np.abs(x)
            if op == OpType.LOG:
                return np.log(np.abs(x) + 1e-8)
            if op == OpType.SIGN:
                return np.sign(x)
            if op in {OpType.TS_MEAN, OpType.TS_STD, OpType.TS_MAX, OpType.TS_MIN, OpType.TS_RANK}:
                return self._ts_op(x, op, node.param or 1)
            if op == OpType.CS_RANK:
                return self._cs_rank(x)
            if op == OpType.CS_ZSCORE:
                return self._cs_zscore(x)

        if op in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}:
            a = self.calculate(node.children[0])
            b = self.calculate(node.children[1])
            if op == OpType.ADD:
                return a + b
            if op == OpType.SUB:
                return a - b
            if op == OpType.MUL:
                return a * b
            if op == OpType.DIV:
                return a / (b + 1e-8)
        return np.full(self.data.shape[:2], np.nan)


class ICCalculator:
    @staticmethod
    def _daily_ic(f: np.ndarray, r: np.ndarray) -> float:
        valid = ~np.isnan(f) & ~np.isnan(r)
        if valid.sum() < 10:
            return np.nan
        fv = f[valid]
        rv = r[valid]
        c = np.corrcoef(fv, rv)[0, 1]
        return c if not np.isnan(c) else np.nan

    @staticmethod
    def calculate_ic(factor_values: np.ndarray, returns: np.ndarray) -> float:
        T = factor_values.shape[0]
        ics = []
        for t in range(T):
            c = ICCalculator._daily_ic(factor_values[t], returns[t])
            if not np.isnan(c):
                ics.append(c)
        return float(np.mean(ics)) if ics else 0.0

    @staticmethod
    def calculate_ic_series(factor_values: np.ndarray, returns: np.ndarray) -> np.ndarray:
        T = factor_values.shape[0]
        series = np.zeros(T)
        for t in range(T):
            c = ICCalculator._daily_ic(factor_values[t], returns[t])
            series[t] = 0.0 if np.isnan(c) else c
        return series

    @staticmethod
    def calculate_rank_ic(factor_values: np.ndarray, returns: np.ndarray) -> float:
        # Spearman by ranking within day
        T, N = factor_values.shape
        vals = []
        for t in range(T):
            f = factor_values[t]
            r = returns[t]
            valid = ~np.isnan(f) & ~np.isnan(r)
            if valid.sum() < 10:
                continue
            fr = f[valid].argsort().argsort().astype(float)
            rr = r[valid].argsort().argsort().astype(float)
            c = np.corrcoef(fr, rr)[0, 1]
            if not np.isnan(c):
                vals.append(c)
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def calculate_icir(factor_values: np.ndarray, returns: np.ndarray) -> float:
        series = ICCalculator.calculate_ic_series(factor_values, returns)
        m = np.mean(series)
        s = np.std(series) + 1e-8
        return float(m / s)
