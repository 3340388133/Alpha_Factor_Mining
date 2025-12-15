"""创新点3: Market Antagonist - min_D max_G [IC - α*VulnScore]"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import random

from ..core.expression import OpType, ASTNode
from ..core.calculator import FactorCalculator, ICCalculator
from ..core.encoder import FactorExpressionEncoder


@dataclass
class AntagonistConfig:
    ast_dim: int = 128
    return_seq_dim: int = 64
    collinearity_dim: int = 32
    hidden_dim: int = 256
    lr_d: float = 1e-4
    alpha: float = 0.5
    d_steps_per_g_step: int = 2
    replay_buffer_size: int = 10000
    batch_size: int = 32
    candidates_per_episode: int = 16
    top_k_evaluate: int = 4


class CollinearityAnalyzer:
    def __init__(self, max_factors: int = 1000, top_k: int = 10):
        self.existing_factors: List[np.ndarray] = []
        self.max_factors = max_factors
        self.top_k = top_k

    def add_factor(self, values: np.ndarray):
        self.existing_factors.append(values.copy())
        if len(self.existing_factors) > self.max_factors:
            self.existing_factors = self.existing_factors[-self.max_factors:]

    def calculate_features(self, new_factor: np.ndarray) -> np.ndarray:
        if not self.existing_factors:
            return np.zeros(32)

        correlations = []
        new_flat = new_factor.flatten()
        new_mask = ~np.isnan(new_flat)

        for existing in self.existing_factors:
            exist_flat = existing.flatten()
            mask = new_mask & ~np.isnan(exist_flat)
            if mask.sum() > 100:
                corr = np.corrcoef(new_flat[mask], exist_flat[mask])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        if not correlations:
            return np.zeros(32)

        correlations = np.array(correlations)
        base = [np.mean(correlations), np.max(correlations), np.std(correlations),
                np.percentile(correlations, 90), np.percentile(correlations, 75),
                np.percentile(correlations, 50), np.sum(correlations > 0.7),
                np.sum(correlations > 0.5), np.sum(correlations > 0.3), len(correlations)]

        sorted_corr = np.sort(correlations)[-self.top_k:]
        if len(sorted_corr) < self.top_k:
            sorted_corr = np.pad(sorted_corr, (self.top_k - len(sorted_corr), 0))

        features = np.array(base + sorted_corr.tolist())
        return np.pad(features, (0, max(0, 32 - len(features))))[:32]


class MarketAntagonist(nn.Module):
    def __init__(self, config: AntagonistConfig = None):
        super().__init__()
        self.config = config or AntagonistConfig()

        self.ast_encoder = nn.Sequential(
            nn.Linear(self.config.ast_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        self.return_encoder = nn.LSTM(1, self.config.return_seq_dim, 2, batch_first=True, dropout=0.2)
        self.return_fc = nn.Linear(self.config.return_seq_dim, self.config.hidden_dim)

        self.collinearity_encoder = nn.Sequential(
            nn.Linear(self.config.collinearity_dim, self.config.hidden_dim // 2),
            nn.ReLU(), nn.Linear(self.config.hidden_dim // 2, self.config.hidden_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim * 2),
            nn.LayerNorm(self.config.hidden_dim * 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )

        self.vulnerability_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, ast_enc, return_seq, collin_feat):
        ast_h = self.ast_encoder(ast_enc)
        _, (ret_h, _) = self.return_encoder(return_seq)
        ret_h = self.return_fc(ret_h[-1])
        col_h = self.collinearity_encoder(collin_feat)
        fused = self.fusion(torch.cat([ast_h, ret_h, col_h], dim=-1))
        return self.vulnerability_head(fused), None


class GANRLTrainer:
    def __init__(self, generator, data: np.ndarray, returns: np.ndarray,
                 feature_names: List[str], config: AntagonistConfig = None):
        self.generator = generator
        self.data = data
        self.returns = returns
        self.feature_names = feature_names
        self.config = config or AntagonistConfig()

        self.calculator = FactorCalculator(data, feature_names)
        self.ast_encoder = FactorExpressionEncoder(len(OpType), len(feature_names), hidden_dim=self.config.ast_dim)
        self.antagonist = MarketAntagonist(self.config)
        self.collinearity_analyzer = CollinearityAnalyzer()
        self.optimizer_d = torch.optim.Adam(self.antagonist.parameters(), lr=self.config.lr_d)
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)
        self.op_to_idx = {op: i for i, op in enumerate(OpType)}
        self.feature_to_idx = {f: i for i, f in enumerate(feature_names)}
        self.train_history = {"g_rewards": [], "d_losses": [], "ic_mean": [], "vuln_mean": [], "collection_ic": []}

    def prepare_input(self, factor: ASTNode, values: np.ndarray) -> Dict:
        with torch.no_grad():
            ast_enc = self.ast_encoder(factor, self.op_to_idx, self.feature_to_idx).squeeze(0)
        returns = np.nanmean(values, axis=1)[-100:]
        if len(returns) < 100:
            returns = np.pad(returns, (100 - len(returns), 0))
        return {"ast_encoding": ast_enc,
                "return_sequence": torch.tensor(returns, dtype=torch.float32).unsqueeze(-1),
                "collinearity_features": torch.tensor(self.collinearity_analyzer.calculate_features(values), dtype=torch.float32)}

    def get_vulnerability(self, factor: ASTNode, values: np.ndarray) -> float:
        self.antagonist.eval()
        inp = self.prepare_input(factor, values)
        with torch.no_grad():
            vuln, _ = self.antagonist(inp["ast_encoding"].unsqueeze(0),
                                      inp["return_sequence"].unsqueeze(0),
                                      inp["collinearity_features"].unsqueeze(0))
        return vuln.item()

    def train_antagonist(self, batch: List[Dict]) -> float:
        self.antagonist.train()
        if len(batch) < 2:
            return 0.0

        ast_list, ret_list, col_list, labels = [], [], [], []
        for s in batch:
            if "inputs" not in s:
                continue
            inp = s["inputs"]
            ast_list.append(inp["ast_encoding"])
            ret_list.append(inp["return_sequence"])
            col_list.append(inp["collinearity_features"])
            orig = abs(s.get("original_ic", 0.001))
            robust = abs(s.get("robust_ic", orig))
            labels.append(1 - (robust / orig if orig > 0.001 else 0))

        if len(ast_list) < 2:
            return 0.0

        ast_b = torch.stack(ast_list)
        ret_b = torch.nn.utils.rnn.pad_sequence(ret_list, batch_first=True)
        col_b = torch.stack(col_list)
        label_b = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        vuln_pred, _ = self.antagonist(ast_b, ret_b, col_b)
        loss = F.mse_loss(vuln_pred, label_b)

        self.optimizer_d.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.antagonist.parameters(), 0.5)
        self.optimizer_d.step()
        return loss.item()

    def train_episode(self) -> Dict:
        candidates = []
        for _ in range(self.config.candidates_per_episode):
            factor, info = self.generator.sample_factor()
            if "error" not in info:
                candidates.append({"factor": factor, "info": info})

        if not candidates:
            return {"error": "No valid factors"}

        candidates.sort(key=lambda x: abs(x["info"]["ic"]), reverse=True)
        eval_set = candidates[: self.config.top_k_evaluate]
        detailed = []
        for c in eval_set:
            values = c["info"]["factor_values"]
            ic = c["info"]["ic"]
            inputs = self.prepare_input(c["factor"], values)
            vuln = self.get_vulnerability(c["factor"], values)

            from .crowding_simulator import CrowdingSimulator
            cs = CrowdingSimulator()
            robust_ic, _, _ = cs.calculate_robust_ic(values, self.returns, use_estimated=False)

            reward = abs(ic) - self.config.alpha * vuln
            sample = {"factor": c["factor"], "info": c["info"], "inputs": inputs, "ic": ic,
                     "original_ic": ic, "robust_ic": robust_ic, "vulnerability_score": vuln, "reward": reward}
            detailed.append(sample)
            self.replay_buffer.append(sample)
        detailed.sort(key=lambda x: x["reward"], reverse=True)
        best = detailed[0]

        if best["reward"] > 0.005:
            self.generator.factor_collection.append(best["factor"])
            self.collinearity_analyzer.add_factor(best["info"]["factor_values"])

        d_loss = 0.0
        if len(self.replay_buffer) >= self.config.batch_size:
            for _ in range(self.config.d_steps_per_g_step):
                batch = random.sample(list(self.replay_buffer), min(self.config.batch_size, len(self.replay_buffer)))
                d_loss += self.train_antagonist(batch)
            d_loss /= self.config.d_steps_per_g_step

        collection_ic, _ = self.generator.calculate_collection_performance()
        self.train_history["g_rewards"].append(best["reward"])
        self.train_history["d_losses"].append(d_loss)
        self.train_history["ic_mean"].append(best["ic"])
        self.train_history["vuln_mean"].append(best["vulnerability_score"])
        self.train_history["collection_ic"].append(collection_ic)

        return {"reward": best["reward"], "ic": best["ic"], "vulnerability_score": best["vulnerability_score"],
                "d_loss": d_loss, "collection_ic": collection_ic, "num_factors": len(self.generator.factor_collection),
                "formula": best["info"]["formula"]}

    def train(self, num_episodes: int = 100, verbose: bool = True) -> Dict:
        for ep in range(num_episodes):
            stats = self.train_episode()
            if verbose and (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{num_episodes} | IC: {stats.get('ic', 0):.4f} | "
                      f"Vuln: {stats.get('vulnerability_score', 0):.4f} | D Loss: {stats.get('d_loss', 0):.4f}")
        return {"history": self.train_history, "num_factors": len(self.generator.factor_collection),
                "final_collection_ic": self.train_history["collection_ic"][-1] if self.train_history["collection_ic"] else 0}

    def get_statistics(self) -> Dict:
        return {"num_episodes": len(self.train_history["g_rewards"]),
                "mean_reward": np.mean(self.train_history["g_rewards"]) if self.train_history["g_rewards"] else 0,
                "mean_ic": np.mean(self.train_history["ic_mean"]) if self.train_history["ic_mean"] else 0,
                "mean_vulnerability": np.mean(self.train_history["vuln_mean"]) if self.train_history["vuln_mean"] else 0}
