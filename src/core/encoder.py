import torch
import torch.nn as nn
from typing import Dict, List

from .expression import ASTNode, OpType


class FactorExpressionEncoder(nn.Module):
    def __init__(self, num_ops: int, num_features: int, hidden_dim: int = 128):
        super().__init__()
        self.num_ops = num_ops
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.op_emb = nn.Embedding(num_ops, hidden_dim // 2)
        self.feat_emb = nn.Embedding(num_features, hidden_dim // 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def _collect(self, node: ASTNode, op_to_idx: Dict[OpType, int], feature_to_idx: Dict[str, int], ops: List[int], feats: List[int]):
        if node.op == OpType.FEATURE:
            idx = feature_to_idx.get(str(node.value), None)
            if idx is not None:
                feats.append(idx)
            return
        if node.op == OpType.CONSTANT:
            return
        ops.append(op_to_idx.get(node.op, 0))
        for c in node.children:
            self._collect(c, op_to_idx, feature_to_idx, ops, feats)

    def forward(self, node: ASTNode, op_to_idx: Dict[OpType, int], feature_to_idx: Dict[str, int]):
        ops: List[int] = []
        feats: List[int] = []
        self._collect(node, op_to_idx, feature_to_idx, ops, feats)

        if not ops:
            ops = [0]
        if not feats:
            feats = [0]

        op_tensor = torch.tensor(ops, dtype=torch.long)
        feat_tensor = torch.tensor(feats, dtype=torch.long)

        op_vec = self.op_emb(op_tensor).mean(dim=0)
        feat_vec = self.feat_emb(feat_tensor).mean(dim=0)

        enc = torch.cat([op_vec, feat_vec], dim=-1)
        enc = self.fc(enc)
        return enc.unsqueeze(0)
