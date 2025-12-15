"""判别器模型 (用于创新点3)"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    因子脆弱性判别器

    用于创新点3: Market Antagonist
    """

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
