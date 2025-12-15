"""
Alpha因子生成的强化学习环境

基于论文框架设计的因子生成环境
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gym
from gym import spaces

from ..core.expression import OpType, ASTNode, ExpressionTree
from ..core.calculator import FactorCalculator, ICCalculator


@dataclass
class AlphaEnvConfig:
    """Alpha生成环境配置"""
    max_depth: int = 6                    # 最大树深度
    max_nodes: int = 20                   # 最大节点数
    window_sizes: List[int] = None        # 时序窗口选项
    constant_range: Tuple[float, float] = (-1.0, 1.0)  # 常数范围
    diversity_bonus: float = 0.1          # 多样性奖励
    complexity_penalty: float = 0.01      # 复杂度惩罚

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [1, 2, 3, 5, 10, 20]


class AlphaGenerationEnv(gym.Env):
    """
    Alpha因子生成环境

    状态：当前部分生成的表达式树
    动作：选择下一个操作符/特征/参数
    奖励：生成完整因子后的IC值（或其他指标）
    """

    def __init__(self,
                 data: np.ndarray,
                 returns: np.ndarray,
                 feature_names: List[str],
                 config: AlphaEnvConfig = None):
        """
        Args:
            data: shape (T, N, F) - 市场数据
            returns: shape (T, N) - 收益率数据
            feature_names: 特征名称列表
            config: 环境配置
        """
        super().__init__()

        self.data = data
        self.returns = returns
        self.feature_names = feature_names
        self.config = config or AlphaEnvConfig()

        # 因子计算器
        self.calculator = FactorCalculator(data, feature_names)

        # 构建动作空间
        self._build_action_space()

        # 当前生成的表达式
        self.current_tree: Optional[ExpressionTree] = None
        self.pending_nodes: List[Tuple[ASTNode, int]] = []  # (父节点, 子索引)
        self.generated_factors: List[ASTNode] = []  # 已生成的因子集合

        # 状态空间 (简化为离散)
        self.observation_space = spaces.Dict({
            "depth": spaces.Discrete(self.config.max_depth + 1),
            "node_count": spaces.Discrete(self.config.max_nodes + 1),
            "pending_count": spaces.Discrete(10),
            "num_factors": spaces.Discrete(100)
        })

    def _build_action_space(self):
        """构建动作空间"""
        # 操作符列表
        self.ops = OpType.get_non_terminal_ops()
        self.terminal_ops = OpType.get_terminal_ops()

        # 动作映射
        self.actions = []

        # 非终止操作符
        for op in self.ops:
            if op.requires_param():
                for window in self.config.window_sizes:
                    self.actions.append(("op", op, window))
            else:
                self.actions.append(("op", op, None))

        # 特征
        for feat in self.feature_names:
            self.actions.append(("feature", feat, None))

        # 常数 (离散化)
        for const in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            self.actions.append(("constant", const, None))

        self.action_space = spaces.Discrete(len(self.actions))

        # 构建索引映射
        self.op_to_idx = {op: i for i, op in enumerate(OpType)}
        self.feature_to_idx = {f: i for i, f in enumerate(self.feature_names)}

    def reset(self) -> Dict:
        """重置环境"""
        self.current_tree = None
        self.pending_nodes = []
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作索引

        Returns:
            (observation, reward, done, info)
        """
        action_type, value, param = self.actions[action]

        # 创建节点
        if action_type == "op":
            node = ASTNode(value, param=param)
            arity = value.get_arity()
        elif action_type == "feature":
            node = ASTNode(OpType.FEATURE, value=value)
            arity = 0
        else:  # constant
            node = ASTNode(OpType.CONSTANT, value=value)
            arity = 0

        # 添加到树中
        if self.current_tree is None:
            # 创建根节点
            self.current_tree = ExpressionTree(node)
            # 添加需要填充的子节点
            for i in range(arity):
                self.pending_nodes.append((node, i))
        else:
            # 填充待处理的节点
            if self.pending_nodes:
                parent, child_idx = self.pending_nodes.pop(0)
                if len(parent.children) <= child_idx:
                    parent.children.extend([None] * (child_idx - len(parent.children) + 1))
                parent.children[child_idx] = node

                # 添加新节点的子节点到待处理列表
                for i in range(arity):
                    self.pending_nodes.append((node, i))

        # 检查是否完成
        done = len(self.pending_nodes) == 0 and self.current_tree is not None

        # 计算奖励
        reward = 0.0
        info = {}

        if done:
            # 因子生成完成，计算奖励
            reward, info = self._calculate_reward()

        # 检查是否超过限制
        if self.current_tree and self.current_tree.root:
            if self.current_tree.root.get_depth() > self.config.max_depth:
                done = True
                reward = -1.0  # 惩罚
            if self.current_tree.root.get_node_count() > self.config.max_nodes:
                done = True
                reward = -1.0

        return self._get_observation(), reward, done, info

    def _calculate_reward(self) -> Tuple[float, Dict]:
        """计算奖励"""
        if self.current_tree is None or self.current_tree.root is None:
            return 0.0, {}

        try:
            # 计算因子值
            factor_values = self.calculator.calculate(self.current_tree.root)

            # 计算IC
            ic = ICCalculator.calculate_ic(factor_values, self.returns)
            rank_ic = ICCalculator.calculate_rank_ic(factor_values, self.returns)
            icir = ICCalculator.calculate_icir(factor_values, self.returns)

            # 基础奖励：IC的绝对值
            reward = abs(ic)

            # 多样性奖励
            diversity_bonus = self._calculate_diversity_bonus(factor_values)
            reward += self.config.diversity_bonus * diversity_bonus

            # 复杂度惩罚
            complexity = self.current_tree.root.get_node_count()
            reward -= self.config.complexity_penalty * complexity

            info = {
                "ic": ic,
                "rank_ic": rank_ic,
                "icir": icir,
                "formula": self.current_tree.to_string(),
                "complexity": complexity,
                "diversity_bonus": diversity_bonus
            }

            return reward, info

        except Exception as e:
            return -0.5, {"error": str(e)}

    def _calculate_diversity_bonus(self, factor_values: np.ndarray) -> float:
        """计算与已有因子的多样性奖励"""
        if not self.generated_factors:
            return 1.0  # 第一个因子，给予满分多样性

        # 计算与所有已有因子的相关性
        correlations = []
        new_flat = factor_values.flatten()
        new_mask = ~np.isnan(new_flat)

        for existing_factor in self.generated_factors:
            existing_values = self.calculator.calculate(existing_factor)
            exist_flat = existing_values.flatten()
            valid_mask = new_mask & ~np.isnan(exist_flat)

            if valid_mask.sum() > 100:
                corr = np.corrcoef(new_flat[valid_mask], exist_flat[valid_mask])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        if not correlations:
            return 1.0

        # 多样性 = 1 - 最大相关性
        max_corr = max(correlations)
        return 1.0 - max_corr

    def _get_observation(self) -> Dict:
        """获取观测"""
        if self.current_tree is None or self.current_tree.root is None:
            depth = 0
            node_count = 0
        else:
            depth = self.current_tree.root.get_depth()
            node_count = self.current_tree.root.get_node_count()

        return {
            "depth": min(depth, self.config.max_depth),
            "node_count": min(node_count, self.config.max_nodes),
            "pending_count": min(len(self.pending_nodes), 9),
            "num_factors": min(len(self.generated_factors), 99)
        }

    def add_factor_to_collection(self, factor: ASTNode):
        """将因子添加到集合中"""
        self.generated_factors.append(factor.copy())

    def get_current_formula(self) -> str:
        """获取当前表达式的公式字符串"""
        if self.current_tree is None:
            return ""
        return self.current_tree.to_string()

    def sample_random_factor(self, max_depth: int = 4) -> ASTNode:
        """
        随机采样一个因子表达式

        用于初始化或探索
        """
        def random_tree(depth: int = 0) -> ASTNode:
            # 到达最大深度或概率性终止
            if depth >= max_depth or (depth > 1 and random.random() < 0.3):
                # 叶节点
                if random.random() < 0.8:
                    return ASTNode(OpType.FEATURE, value=random.choice(self.feature_names))
                else:
                    const_val = random.uniform(*self.config.constant_range)
                    return ASTNode(OpType.CONSTANT, value=const_val)

            # 选择操作符
            op = random.choice(self.ops)
            arity = op.get_arity()

            # 时序参数
            param = None
            if op.requires_param():
                param = random.choice(self.config.window_sizes)

            # 递归生成子节点
            children = [random_tree(depth + 1) for _ in range(arity)]

            return ASTNode(op, children=children, param=param)

        return random_tree()


class SynergisticAlphaEnv(AlphaGenerationEnv):
    """
    协同Alpha因子生成环境

    扩展基础环境，增加因子集合协同性的考量
    基于论文 "Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning"
    """

    def __init__(self,
                 data: np.ndarray,
                 returns: np.ndarray,
                 feature_names: List[str],
                 config: AlphaEnvConfig = None,
                 combination_model: str = "linear"):
        """
        Args:
            combination_model: 组合模型类型 ("linear", "ridge", "xgboost")
        """
        super().__init__(data, returns, feature_names, config)
        self.combination_model = combination_model
        self.factor_values_cache: Dict[str, np.ndarray] = {}

    def _calculate_reward(self) -> Tuple[float, Dict]:
        """
        计算奖励

        论文核心：使用下游组合模型的表现作为奖励
        """
        if self.current_tree is None or self.current_tree.root is None:
            return 0.0, {}

        try:
            # 计算新因子值
            new_factor_values = self.calculator.calculate(self.current_tree.root)

            # 单因子IC
            ic = ICCalculator.calculate_ic(new_factor_values, self.returns)

            # 计算组合模型的增量贡献
            synergy_score = self._calculate_synergy_contribution(new_factor_values)

            # 最终奖励 = IC + 协同贡献
            reward = abs(ic) + 0.5 * synergy_score

            # 复杂度惩罚
            complexity = self.current_tree.root.get_node_count()
            reward -= self.config.complexity_penalty * complexity

            info = {
                "ic": ic,
                "synergy_score": synergy_score,
                "formula": self.current_tree.to_string(),
                "complexity": complexity
            }

            return reward, info

        except Exception as e:
            return -0.5, {"error": str(e)}

    def _calculate_synergy_contribution(self, new_factor_values: np.ndarray) -> float:
        """
        计算新因子对组合的协同贡献

        论文核心方法：
        1. 使用当前因子集合构建预测模型
        2. 加入新因子后重新构建模型
        3. 计算预测能力的提升
        """
        if not self.generated_factors:
            return 0.5  # 第一个因子，给予基础分

        # 获取所有已有因子的值
        existing_factor_matrix = []
        for factor in self.generated_factors:
            formula = factor.to_string()
            if formula in self.factor_values_cache:
                values = self.factor_values_cache[formula]
            else:
                values = self.calculator.calculate(factor)
                self.factor_values_cache[formula] = values
            existing_factor_matrix.append(values)

        # 构建特征矩阵
        T, N = new_factor_values.shape

        # 原有因子组合的IC (使用简单平均作为组合)
        old_combined = np.mean(existing_factor_matrix, axis=0)
        old_ic = abs(ICCalculator.calculate_ic(old_combined, self.returns))

        # 加入新因子后的组合IC
        all_factors = existing_factor_matrix + [new_factor_values]
        new_combined = np.mean(all_factors, axis=0)
        new_ic = abs(ICCalculator.calculate_ic(new_combined, self.returns))

        # 协同贡献 = IC提升
        synergy = new_ic - old_ic

        # 正向贡献给予奖励，负向给予惩罚
        return max(synergy, -0.5)
