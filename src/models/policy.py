"""策略网络模型"""

import torch
import torch.nn as nn
from typing import Tuple


class PolicyNetwork(nn.Module):
    """因子生成策略网络"""

    def __init__(self,
                 num_ops: int,
                 num_features: int,
                 num_params: int,
                 hidden_dim: int = 256):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # 动作头
        self.op_head = nn.Linear(hidden_dim, num_ops)
        self.feature_head = nn.Linear(hidden_dim, num_features)
        self.param_head = nn.Linear(hidden_dim, num_params)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, hidden: Tuple = None):
        if hidden is None:
            out, hidden = self.lstm(state.unsqueeze(1))
        else:
            out, hidden = self.lstm(state.unsqueeze(1), hidden)

        out = out.squeeze(1)

        return (
            self.op_head(out),
            self.feature_head(out),
            self.param_head(out),
            self.value_head(out),
            hidden
        )


class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
