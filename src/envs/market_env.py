"""
股票市场仿真环境

提供股票交易的仿真环境，用于评估因子的投资表现
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    """市场配置"""
    initial_capital: float = 1_000_000.0  # 初始资金
    commission_rate: float = 0.001         # 交易佣金率
    slippage_rate: float = 0.001           # 滑点率
    max_position_ratio: float = 0.1        # 单只股票最大持仓比例
    risk_free_rate: float = 0.03           # 无风险利率(年化)
    trading_days_per_year: int = 252       # 每年交易日


@dataclass
class Position:
    """持仓信息"""
    symbol: int           # 股票索引
    shares: float         # 持仓股数
    avg_cost: float       # 平均成本
    current_price: float  # 当前价格
    unrealized_pnl: float = 0.0  # 未实现盈亏

    @property
    def market_value(self) -> float:
        """市值"""
        return self.shares * self.current_price


@dataclass
class TradeRecord:
    """交易记录"""
    time_step: int
    symbol: int
    action: str  # 'buy' or 'sell'
    shares: float
    price: float
    commission: float
    slippage: float


class StockMarketEnv:
    """
    股票市场仿真环境

    支持多股票投资组合的仿真交易
    """

    def __init__(self,
                 prices: np.ndarray,
                 returns: np.ndarray,
                 config: MarketConfig = None):
        """
        Args:
            prices: shape (T, N) - 股票价格序列
            returns: shape (T, N) - 股票收益率序列
            config: 市场配置
        """
        self.prices = prices
        self.returns = returns
        self.T, self.N = prices.shape
        self.config = config or MarketConfig()

        # 状态变量
        self.current_step = 0
        self.cash = self.config.initial_capital
        self.positions: Dict[int, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.portfolio_values: List[float] = []

        # 记录
        self.daily_returns: List[float] = []

    def reset(self, start_step: int = 0) -> Dict:
        """
        重置环境

        Args:
            start_step: 起始时间步

        Returns:
            初始观测
        """
        self.current_step = start_step
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_values = [self.config.initial_capital]
        self.daily_returns = []

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步交易

        Args:
            action: shape (N,) - 目标持仓权重 (sum = 1)

        Returns:
            (observation, reward, done, info)
        """
        # 记录执行前的组合价值
        prev_portfolio_value = self._get_portfolio_value()

        # 执行交易
        self._execute_trades(action)

        # 时间步进
        self.current_step += 1

        # 更新持仓价格
        self._update_positions()

        # 计算新的组合价值
        new_portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(new_portfolio_value)

        # 计算收益率
        daily_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.daily_returns.append(daily_return)

        # 计算奖励 (可以是收益率、夏普率等)
        reward = daily_return

        # 检查是否结束
        done = self.current_step >= self.T - 1

        # 构建info
        info = {
            "portfolio_value": new_portfolio_value,
            "daily_return": daily_return,
            "cash": self.cash,
            "num_positions": len(self.positions)
        }

        return self._get_observation(), reward, done, info

    def _execute_trades(self, target_weights: np.ndarray):
        """执行交易到目标权重"""
        # 归一化权重
        target_weights = np.clip(target_weights, 0, 1)
        weight_sum = target_weights.sum()
        if weight_sum > 0:
            target_weights = target_weights / weight_sum
        else:
            target_weights = np.zeros(self.N)

        # 当前价格
        current_prices = self.prices[self.current_step]

        # 当前组合总价值
        total_value = self._get_portfolio_value()

        # 计算目标持仓金额
        target_values = target_weights * total_value

        # 先卖出需要减仓的股票
        for symbol in list(self.positions.keys()):
            if symbol >= self.N:
                continue

            current_value = self.positions[symbol].market_value
            target_value = target_values[symbol]

            if target_value < current_value:
                # 需要卖出
                sell_value = current_value - target_value
                sell_shares = sell_value / current_prices[symbol]
                self._sell(symbol, sell_shares, current_prices[symbol])

        # 再买入需要加仓的股票
        for symbol in range(self.N):
            if np.isnan(current_prices[symbol]) or current_prices[symbol] <= 0:
                continue

            current_value = self.positions[symbol].market_value if symbol in self.positions else 0
            target_value = target_values[symbol]

            if target_value > current_value:
                # 需要买入
                buy_value = min(target_value - current_value, self.cash * 0.99)  # 保留1%现金
                buy_shares = buy_value / current_prices[symbol]
                if buy_shares > 0:
                    self._buy(symbol, buy_shares, current_prices[symbol])

    def _buy(self, symbol: int, shares: float, price: float):
        """买入"""
        if shares <= 0 or price <= 0:
            return

        # 计算成本
        gross_cost = shares * price
        commission = gross_cost * self.config.commission_rate
        slippage = gross_cost * self.config.slippage_rate
        total_cost = gross_cost + commission + slippage

        if total_cost > self.cash:
            # 资金不足，调整买入数量
            shares = (self.cash / (1 + self.config.commission_rate + self.config.slippage_rate)) / price
            gross_cost = shares * price
            commission = gross_cost * self.config.commission_rate
            slippage = gross_cost * self.config.slippage_rate
            total_cost = gross_cost + commission + slippage

        if shares <= 0:
            return

        # 更新现金
        self.cash -= total_cost

        # 更新持仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
            pos.current_price = price
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=price,
                current_price=price
            )

        # 记录交易
        self.trade_history.append(TradeRecord(
            time_step=self.current_step,
            symbol=symbol,
            action='buy',
            shares=shares,
            price=price,
            commission=commission,
            slippage=slippage
        ))

    def _sell(self, symbol: int, shares: float, price: float):
        """卖出"""
        if symbol not in self.positions or shares <= 0:
            return

        pos = self.positions[symbol]
        shares = min(shares, pos.shares)

        if shares <= 0:
            return

        # 计算收入
        gross_revenue = shares * price
        commission = gross_revenue * self.config.commission_rate
        slippage = gross_revenue * self.config.slippage_rate
        net_revenue = gross_revenue - commission - slippage

        # 更新现金
        self.cash += net_revenue

        # 更新持仓
        pos.shares -= shares
        if pos.shares < 1e-8:
            del self.positions[symbol]
        else:
            pos.current_price = price

        # 记录交易
        self.trade_history.append(TradeRecord(
            time_step=self.current_step,
            symbol=symbol,
            action='sell',
            shares=shares,
            price=price,
            commission=commission,
            slippage=slippage
        ))

    def _update_positions(self):
        """更新持仓价格"""
        if self.current_step >= self.T:
            return

        current_prices = self.prices[self.current_step]

        for symbol, pos in list(self.positions.items()):
            if symbol < len(current_prices):
                new_price = current_prices[symbol]
                if not np.isnan(new_price) and new_price > 0:
                    pos.current_price = new_price
                    pos.unrealized_pnl = (pos.current_price - pos.avg_cost) * pos.shares

    def _get_portfolio_value(self) -> float:
        """获取当前组合总价值"""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value

    def _get_observation(self) -> Dict:
        """获取当前观测"""
        return {
            "step": self.current_step,
            "prices": self.prices[self.current_step].copy(),
            "returns": self.returns[self.current_step].copy() if self.current_step > 0 else np.zeros(self.N),
            "cash": self.cash,
            "portfolio_value": self._get_portfolio_value(),
            "positions": {s: (p.shares, p.current_price) for s, p in self.positions.items()}
        }

    def get_performance_metrics(self) -> Dict:
        """获取回测绩效指标"""
        if len(self.portfolio_values) < 2:
            return {}

        values = np.array(self.portfolio_values)
        returns = np.array(self.daily_returns)

        # 总收益率
        total_return = (values[-1] - values[0]) / values[0]

        # 年化收益率
        n_days = len(returns)
        annual_return = (1 + total_return) ** (self.config.trading_days_per_year / n_days) - 1

        # 年化波动率
        annual_volatility = np.std(returns) * np.sqrt(self.config.trading_days_per_year)

        # 夏普比率
        excess_return = annual_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)

        # 卡尔玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 交易统计
        total_trades = len(self.trade_history)
        total_commission = sum(t.commission for t in self.trade_history)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "total_trades": total_trades,
            "total_commission": total_commission,
            "final_value": values[-1]
        }


class FactorBasedStrategy:
    """
    基于因子的投资策略

    根据因子值构建投资组合
    """

    def __init__(self,
                 top_k: int = 10,
                 bottom_k: int = 10,
                 long_only: bool = True,
                 holding_period: int = 1):
        """
        Args:
            top_k: 做多的股票数量
            bottom_k: 做空的股票数量
            long_only: 是否只做多
            holding_period: 持仓周期(天)
        """
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.long_only = long_only
        self.holding_period = holding_period

    def get_weights(self, factor_values: np.ndarray) -> np.ndarray:
        """
        根据因子值获取目标权重

        Args:
            factor_values: shape (N,) - 当期因子值

        Returns:
            shape (N,) - 目标权重
        """
        N = len(factor_values)
        weights = np.zeros(N)

        # 处理NaN
        valid_mask = ~np.isnan(factor_values)
        if valid_mask.sum() < self.top_k:
            return weights

        # 排序
        ranks = np.argsort(factor_values)

        # 做多排名靠前的股票
        top_indices = ranks[-self.top_k:]
        top_indices = top_indices[valid_mask[top_indices]]

        if len(top_indices) > 0:
            weights[top_indices] = 1.0 / len(top_indices)

        if not self.long_only and self.bottom_k > 0:
            # 做空排名靠后的股票 (这里简化为不分配权重)
            pass

        return weights

    def backtest(self,
                 factor_values: np.ndarray,
                 prices: np.ndarray,
                 returns: np.ndarray) -> Dict:
        """
        对因子进行回测

        Args:
            factor_values: shape (T, N) - 因子值序列
            prices: shape (T, N) - 价格序列
            returns: shape (T, N) - 收益率序列

        Returns:
            回测结果
        """
        T, N = factor_values.shape
        env = StockMarketEnv(prices, returns)
        env.reset()

        for t in range(T - 1):
            # 每隔holding_period调仓一次
            if t % self.holding_period == 0:
                weights = self.get_weights(factor_values[t])
            else:
                # 保持原有持仓
                weights = np.zeros(N)
                for symbol, pos in env.positions.items():
                    weights[symbol] = pos.market_value

                total = weights.sum()
                if total > 0:
                    weights /= total

            env.step(weights)

        return env.get_performance_metrics()
