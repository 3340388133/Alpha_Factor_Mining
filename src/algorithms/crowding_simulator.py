"""创新点2: Crowding Simulator - IC_N = IC_0 * exp(-k*N)"""
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from ..core.expression import ASTNode
from ..core.calculator import ICCalculator


@dataclass
class CrowdingConfig:
    k_init: float = 0.1
    k_adaptive: bool = True
    n_max: int = 100
    similarity_threshold: float = 0.7
    history_size: int = 1000
    candidates_per_episode: int = 16
    top_k_evaluate: int = 4


class CrowdingSimulator:
    def __init__(self, config: CrowdingConfig = None):
        self.config = config or CrowdingConfig()
        self.k = self.config.k_init
        self.historical_factors: List[Tuple[ASTNode, np.ndarray, float]] = []
        self.k_history: List[float] = []

    def factor_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        f1, f2 = v1.flatten(), v2.flatten()
        mask = ~(np.isnan(f1) | np.isnan(f2))
        if mask.sum() < 100:
            return 0.0
        corr = np.corrcoef(f1[mask], f2[mask])[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    def estimate_crowding_level(self, factor_values: np.ndarray) -> float:
        score = 0.0
        for _, hist_values, _ in self.historical_factors:
            sim = self.factor_similarity(factor_values, hist_values)
            if sim > self.config.similarity_threshold:
                score += (sim - self.config.similarity_threshold) / (1 - self.config.similarity_threshold)
        return min(score, self.config.n_max)

    def simulate_decay(self, ic_0: float, n: int) -> float:
        return ic_0 * np.exp(-self.k * n)

    def calculate_robust_ic(self, factor_values: np.ndarray, returns: np.ndarray,
                           use_estimated: bool = True) -> Tuple[float, float, Dict]:
        ic_0 = ICCalculator.calculate_ic(factor_values, returns)

        if use_estimated:
            est_n = self.estimate_crowding_level(factor_values)
            ic_sum = sum(self.simulate_decay(ic_0, n) for n in range(int(est_n), self.config.n_max + 1))
            count = self.config.n_max - int(est_n) + 1
            robust_ic = ic_sum / count if count > 0 else ic_0 * 0.1
        else:
            ic_sum = sum(self.simulate_decay(ic_0, n) for n in range(self.config.n_max + 1))
            robust_ic = ic_sum / (self.config.n_max + 1)
            est_n = 0

        decay = robust_ic / ic_0 if abs(ic_0) > 1e-8 else 0
        return robust_ic, ic_0, {"original_ic": ic_0, "robust_ic": robust_ic,
                                 "estimated_crowding": est_n, "decay_factor": decay, "k": self.k}

    def add_to_history(self, factor: ASTNode, values: np.ndarray, ic: float):
        self.historical_factors.append((factor, values.copy(), ic))
        if len(self.historical_factors) > self.config.history_size:
            self.historical_factors = self.historical_factors[-self.config.history_size:]

    def get_statistics(self) -> Dict:
        return {"num_historical": len(self.historical_factors), "k": self.k}


class CrowdingAwareAlphaRL:
    def __init__(self, base_algorithm, crowding_config: CrowdingConfig = None):
        self.base = base_algorithm
        self.crowding = CrowdingSimulator(crowding_config)
        self.train_history = {"episode_rewards": [], "robust_ic": [], "original_ic": [], "decay_factor": []}

    def train_episode(self) -> Dict:
        candidates = []
        for _ in range(self.crowding.config.candidates_per_episode):
            factor, info = self.base.sample_factor()
            if "error" not in info:
                candidates.append({"factor": factor, "info": info})

        if not candidates:
            return {"error": "No valid factors"}

        candidates.sort(key=lambda x: abs(x["info"]["ic"]), reverse=True)
        eval_set = candidates[: self.crowding.config.top_k_evaluate]
        scored = []
        for c in eval_set:
            robust_ic, orig_ic, cr_info = self.crowding.calculate_robust_ic(
                c["info"]["factor_values"], self.base.returns)
            reward = abs(robust_ic) + 0.3 * cr_info["decay_factor"]
            scored.append({"factor": c["factor"], "info": c["info"], "crowding_info": cr_info, "reward": reward})
        scored.sort(key=lambda x: x["reward"], reverse=True)
        best = scored[0]

        if best["reward"] > 0.005:
            self.base.factor_collection.append(best["factor"])
            self.crowding.add_to_history(best["factor"], best["info"]["factor_values"],
                                        best["crowding_info"]["original_ic"])

        self.train_history["episode_rewards"].append(best["reward"])
        self.train_history["robust_ic"].append(best["crowding_info"]["robust_ic"])
        self.train_history["original_ic"].append(best["crowding_info"]["original_ic"])
        self.train_history["decay_factor"].append(best["crowding_info"]["decay_factor"])

        return {"reward": best["reward"], "original_ic": best["crowding_info"]["original_ic"],
                "robust_ic": best["crowding_info"]["robust_ic"],
                "decay_factor": best["crowding_info"]["decay_factor"],
                "formula": best["info"]["formula"]}

    def train(self, num_episodes: int = 100, verbose: bool = True) -> Dict:
        for ep in range(num_episodes):
            stats = self.train_episode()
            if verbose and (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{num_episodes} | Orig IC: {stats.get('original_ic', 0):.4f} | "
                      f"Robust IC: {stats.get('robust_ic', 0):.4f} | Decay: {stats.get('decay_factor', 0):.4f}")
        return {"history": self.train_history, "crowding_stats": self.crowding.get_statistics(),
                "num_factors": len(self.base.factor_collection)}
