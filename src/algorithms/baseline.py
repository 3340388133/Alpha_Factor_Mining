"""Baseline: Synergistic Alpha RL (KDD'23)"""
import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from ..core.expression import OpType, ASTNode
from ..core.calculator import FactorCalculator, ICCalculator


@dataclass
class BaselineConfig:
    hidden_dim: int = 256
    lr: float = 1e-4
    max_depth: int = 5
    max_nodes: int = 15
    window_sizes: List[int] = None
    combination_type: str = "linear"
    candidates_per_episode: int = 16
    top_k_evaluate: int = 4

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [1, 2, 3, 5, 10, 20]


class CombinationModel:
    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type
        self.weights = None

    def fit(self, factor_matrix: np.ndarray, returns: np.ndarray) -> np.ndarray:
        T, N, K = factor_matrix.shape
        if self.model_type == "linear":
            ics = [ICCalculator.calculate_ic(factor_matrix[:, :, k], returns) for k in range(K)]
            abs_ics = np.abs(ics)
            self.weights = abs_ics / abs_ics.sum() if abs_ics.sum() > 0 else np.ones(K) / K
        else:
            self.weights = np.ones(K) / K
        return np.sum(factor_matrix * self.weights.reshape(1, 1, -1), axis=2)


class SynergisticAlphaRL:
    def __init__(self, data: np.ndarray, returns: np.ndarray,
                 feature_names: List[str], config: BaselineConfig = None):
        self.data = data
        self.returns = returns
        self.feature_names = feature_names
        self.config = config or BaselineConfig()
        self.calculator = FactorCalculator(data, feature_names)
        self.combination_model = CombinationModel(self.config.combination_type)
        self.factor_collection: List[ASTNode] = []
        self.factor_values_cache: Dict[str, np.ndarray] = {}
        self.ops = OpType.get_non_terminal_ops()
        self.train_history = {"episode_rewards": [], "collection_ic": [], "num_factors": []}

    def _random_tree(self, depth: int = 0, max_depth: int = 4) -> ASTNode:
        if depth >= max_depth or (depth > 1 and random.random() < 0.3):
            if random.random() < 0.8:
                return ASTNode(OpType.FEATURE, value=random.choice(self.feature_names))
            return ASTNode(OpType.CONSTANT, value=random.uniform(-1, 1))

        op = random.choice(self.ops)
        param = random.choice(self.config.window_sizes) if op.requires_param() else None
        children = [self._random_tree(depth + 1, max_depth) for _ in range(op.get_arity())]
        return ASTNode(op, children=children, param=param)

    def sample_factor(self) -> Tuple[ASTNode, Dict]:
        factor = self._random_tree(max_depth=self.config.max_depth)
        try:
            values = self.calculator.calculate(factor)
            ic = ICCalculator.calculate_ic(values, self.returns)
            return factor, {"factor_values": values, "ic": ic, "formula": factor.to_string(),
                          "depth": factor.get_depth(), "complexity": factor.get_node_count()}
        except:
            return factor, {"error": "calc failed"}

    def calculate_collection_performance(self) -> Tuple[float, np.ndarray]:
        if not self.factor_collection:
            return 0.0, np.zeros_like(self.returns)

        T, N = self.returns.shape
        K = len(self.factor_collection)
        matrix = np.zeros((T, N, K))

        for k, f in enumerate(self.factor_collection):
            formula = f.to_string()
            if formula in self.factor_values_cache:
                matrix[:, :, k] = self.factor_values_cache[formula]
            else:
                values = self.calculator.calculate(f)
                self.factor_values_cache[formula] = values
                matrix[:, :, k] = values

        combined = self.combination_model.fit(matrix, self.returns)
        return ICCalculator.calculate_ic(combined, self.returns), combined

    def calculate_marginal_contribution(self, new_factor: ASTNode) -> float:
        current_ic, _ = self.calculate_collection_performance()
        new_values = self.calculator.calculate(new_factor)
        self.factor_collection.append(new_factor)
        self.factor_values_cache[new_factor.to_string()] = new_values
        new_ic, _ = self.calculate_collection_performance()
        self.factor_collection.pop()
        return new_ic - current_ic

    def train_episode(self) -> Dict:
        candidates = []
        for _ in range(self.config.candidates_per_episode):
            factor, info = self.sample_factor()
            if "error" not in info:
                candidates.append({"factor": factor, "info": info})

        if not candidates:
            return {"error": "No valid factors"}

        candidates.sort(key=lambda x: abs(x["info"]["ic"]), reverse=True)
        eval_set = candidates[: self.config.top_k_evaluate]
        scored = []
        for c in eval_set:
            contrib = self.calculate_marginal_contribution(c["factor"])
            reward = abs(c["info"]["ic"]) + 0.5 * contrib
            scored.append({"factor": c["factor"], "info": c["info"],
                           "marginal_contribution": contrib, "reward": reward})
        scored.sort(key=lambda x: x["reward"], reverse=True)
        best = scored[0]

        if best["reward"] > 0.01:
            self.factor_collection.append(best["factor"])
            self.factor_values_cache[best["factor"].to_string()] = best["info"]["factor_values"]

        collection_ic, _ = self.calculate_collection_performance()
        self.train_history["episode_rewards"].append(best["reward"])
        self.train_history["collection_ic"].append(collection_ic)
        self.train_history["num_factors"].append(len(self.factor_collection))

        return {"reward": best["reward"], "ic": best["info"]["ic"],
                "marginal_contribution": best["marginal_contribution"],
                "collection_ic": collection_ic, "num_factors": len(self.factor_collection),
                "best_formula": best["info"]["formula"]}

    def train(self, num_episodes: int = 100, verbose: bool = True) -> Dict:
        for ep in range(num_episodes):
            stats = self.train_episode()
            if verbose and (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{num_episodes} | Reward: {stats.get('reward', 0):.4f} | "
                      f"Collection IC: {stats.get('collection_ic', 0):.4f} | Factors: {stats.get('num_factors', 0)}")
        return {"final_collection_ic": self.train_history["collection_ic"][-1] if self.train_history["collection_ic"] else 0,
                "num_factors": len(self.factor_collection), "history": self.train_history}

    def get_factor_collection(self) -> List[Dict]:
        return [{"formula": f.to_string(),
                 "ic": ICCalculator.calculate_ic(self.factor_values_cache.get(f.to_string(), np.zeros_like(self.returns)), self.returns),
                 "depth": f.get_depth(), "complexity": f.get_node_count()}
                for f in self.factor_collection]
