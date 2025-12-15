"""
评估指标

用于评估因子和投资组合的表现
"""

import numpy as np
from typing import Dict, List
from ..core.calculator import ICCalculator


def calculate_metrics(
    factor_values: np.ndarray,
    returns: np.ndarray,
    prices: np.ndarray = None
) -> Dict:
    """
    计算因子的综合评估指标

    Args:
        factor_values: shape (T, N) 因子值
        returns: shape (T, N) 收益率
        prices: shape (T, N) 价格 (可选，用于回测)

    Returns:
        指标字典
    """
    metrics = {}

    # IC相关指标
    metrics["ic"] = ICCalculator.calculate_ic(factor_values, returns)
    metrics["rank_ic"] = ICCalculator.calculate_rank_ic(factor_values, returns)
    metrics["icir"] = ICCalculator.calculate_icir(factor_values, returns)

    # IC时间序列
    ic_series = ICCalculator.calculate_ic_series(factor_values, returns)
    metrics["ic_mean"] = np.mean(ic_series)
    metrics["ic_std"] = np.std(ic_series)
    metrics["ic_positive_ratio"] = np.mean(ic_series > 0)

    # 因子稳定性
    metrics["factor_autocorr"] = calculate_factor_autocorrelation(factor_values)

    # 因子覆盖度
    valid_ratio = (~np.isnan(factor_values)).mean()
    metrics["coverage"] = valid_ratio

    # 因子分布
    flat_values = factor_values.flatten()
    valid_values = flat_values[~np.isnan(flat_values)]
    if len(valid_values) > 0:
        metrics["factor_mean"] = np.mean(valid_values)
        metrics["factor_std"] = np.std(valid_values)
        metrics["factor_skew"] = calculate_skewness(valid_values)
        metrics["factor_kurt"] = calculate_kurtosis(valid_values)

    return metrics


def calculate_factor_autocorrelation(factor_values: np.ndarray, lag: int = 1) -> float:
    """
    计算因子的自相关性

    衡量因子值在时间上的稳定性
    """
    T, N = factor_values.shape
    if T <= lag:
        return 0.0

    correlations = []
    for n in range(N):
        series = factor_values[:, n]
        valid = ~np.isnan(series[:-lag]) & ~np.isnan(series[lag:])
        if valid.sum() > 10:
            corr = np.corrcoef(series[:-lag][valid], series[lag:][valid])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def calculate_skewness(x: np.ndarray) -> float:
    """计算偏度"""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


def calculate_kurtosis(x: np.ndarray) -> float:
    """计算峰度"""
    n = len(x)
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return 0.0
    return np.mean(((x - mean) / std) ** 4) - 3


def compare_methods(results: Dict[str, Dict]) -> Dict:
    """
    比较不同方法的结果

    Args:
        results: {方法名: 结果字典}

    Returns:
        比较结果
    """
    comparison = {
        "methods": list(results.keys()),
        "metrics": {}
    }

    # 提取所有方法的共同指标
    all_metrics = set()
    for method_results in results.values():
        if isinstance(method_results, dict):
            all_metrics.update(method_results.keys())

    for metric in all_metrics:
        comparison["metrics"][metric] = {}
        for method, method_results in results.items():
            if isinstance(method_results, dict) and metric in method_results:
                comparison["metrics"][metric][method] = method_results[metric]

    # 计算排名
    comparison["rankings"] = {}
    for metric, values in comparison["metrics"].items():
        if values:
            # 假设指标越大越好 (可以根据实际情况调整)
            sorted_methods = sorted(values.items(), key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric] = [m[0] for m in sorted_methods]

    return comparison
