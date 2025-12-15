from enum import Enum
from typing import List, Optional


class OpType(Enum):
    FEATURE = "FEATURE"
    CONSTANT = "CONSTANT"

    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"

    ABS = "ABS"
    LOG = "LOG"
    SIGN = "SIGN"

    TS_MEAN = "TS_MEAN"
    TS_STD = "TS_STD"
    TS_MAX = "TS_MAX"
    TS_MIN = "TS_MIN"
    TS_RANK = "TS_RANK"

    CS_RANK = "CS_RANK"
    CS_ZSCORE = "CS_ZSCORE"

    def requires_param(self) -> bool:
        return self in {
            OpType.TS_MEAN,
            OpType.TS_STD,
            OpType.TS_MAX,
            OpType.TS_MIN,
            OpType.TS_RANK,
        }

    def get_arity(self) -> int:
        if self in {OpType.FEATURE, OpType.CONSTANT}:
            return 0
        if self in {OpType.ABS, OpType.LOG, OpType.SIGN, OpType.TS_MEAN, OpType.TS_STD, OpType.TS_MAX, OpType.TS_MIN, OpType.TS_RANK,
                    OpType.CS_RANK, OpType.CS_ZSCORE}:
            return 1
        if self in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}:
            return 2
        return 0

    @staticmethod
    def get_non_terminal_ops() -> List["OpType"]:
        return [
            OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
            OpType.ABS, OpType.LOG, OpType.SIGN,
            OpType.TS_MEAN, OpType.TS_STD, OpType.TS_MAX, OpType.TS_MIN, OpType.TS_RANK,
            OpType.CS_RANK, OpType.CS_ZSCORE,
        ]


class ASTNode:
    def __init__(self, op: OpType, value: Optional[float] = None, children: Optional[List["ASTNode"]] = None, param: Optional[int] = None):
        self.op = op
        self.value = value
        self.children = children or []
        self.param = param

    def to_string(self) -> str:
        if self.op == OpType.FEATURE:
            return str(self.value)
        if self.op == OpType.CONSTANT:
            return f"{float(self.value):.3f}"
        if self.op in {OpType.ABS, OpType.LOG, OpType.SIGN, OpType.CS_RANK, OpType.CS_ZSCORE}:
            return f"{self.op.name}({self.children[0].to_string()})"
        if self.op in {OpType.TS_MEAN, OpType.TS_STD, OpType.TS_MAX, OpType.TS_MIN, OpType.TS_RANK}:
            w = self.param or 1
            return f"{self.op.name}({self.children[0].to_string()}, {w})"
        if self.op in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}:
            return f"({self.children[0].to_string()} {self.op.name} {self.children[1].to_string()})"
        return self.op.name

    def get_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.get_depth() for c in self.children)

    def get_node_count(self) -> int:
        return 1 + sum(c.get_node_count() for c in self.children)
