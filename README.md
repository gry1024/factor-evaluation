# 单因子评价系统

基于 [量化回测系统](https://github.com/gry1024/Backtest-System-for-QR) 的行情、表达式引擎和真实撮合回测，对单因子做预测能力与组合表现评价。第一版聚焦单因子研究，不含因子组合优化。

## 评价内容

- **预测能力**：每日 Pearson IC、Spearman RankIC、ICIR / RankICIR、IC 正值比例与有效交易日。
- **分层选股**：按每日因子分位数分为五组，统计未来 20 日开盘到开盘收益、最高/最低组差值及各组胜率。此项是重叠持有期的预测诊断，不按日频 Sharpe 年化。
- **组合回测**：调用原回测引擎的真实日频模拟，计入佣金和卖出印花税，输出年化收益、Sharpe、最大回撤、Calmar、信息比率、换手率、基准超额收益。
- **样本说明**：使用历史每日非 ST 股票池；因子收盘生成、次日开盘交易。回测区间、撮合、成本及数据哈希由上游引擎记录。

## 首批评价因子

| 因子 | 表达式来源 | 上游回测年化收益 | Sharpe | RankIC |
|---|---|---:|---:|---:|
| 非流动性 / 价格冲击 | `TsMean(Div(Abs(TsPctChange($close,2)),Add($amt,1)),20)` | 41.46% | 1.570 | 0.0903 |
| 低波动 | `Mul(-1,TsStd(TsPctChange($close,2),20))` | 15.92% | 0.971 | 0.0664 |
| 收益偏度 | `Mul(-1,TsSkew(TsPctChange($close,2),20))` | 19.34% | 0.922 | 0.0185 |
| 日内反转 | 上游单因子实验配置 | 17.10% | 0.709 | 0.0230 |

数据来自上游 `showcase/all_experiments.csv`，统一区间为 2025-01-02 至 2026-09-30。非流动性因子采用成本后组合 Sharpe 排名第一；完整现成回测结果见上游项目。新跑出的 IC/分层结果由本项目直接从 Feather 行情重算。

## 快速开始

要求 Python 3.11+，并已准备上游回测项目及其本地 Feather 行情、历史状态和指数数据。原始行情不包含在本仓库中。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m factoreval --backtest-root "..\quant-backtest-system" --output reports
```

也可显式传入路径：

```powershell
python -m factoreval `
  --backtest-root "E:\path\to\quant-backtest-system" `
  --output reports
```

程序会使用上游 `quantbt` 表达式解析、行情加载、可交易股票过滤、标签和交易撮合逻辑；每个因子输出 JSON 指标、逐日 IC、分层收益 CSV，并生成便于阅读的 Markdown 汇总。上游行情目录默认是 `quant-backtest-system/feathers`，状态与指数文件分别位于其 `market_status/state.feather` 和 `benchmarks/csi300.csv`。

## 输出结构

```text
reports/
  summary.md
  summary.csv
  illiquidity/
    metrics.json
    daily_ic.csv
    quantile_returns.csv
    backtest/       # 与上游引擎相同的完整可审计回测结果
```

## 口径与限制

分层收益是信号日后 20 个交易日的开盘到开盘收益，组间收益存在重叠，因此主要用于观察单调性；实际策略收益和 Sharpe 使用上游逐日持仓模拟，避免对重叠标签直接年化。样本较短且覆盖单一市场阶段，因子表现不构成未来收益承诺。因子符号沿用上游配置：高分代表更优先买入。

## 项目结构

`factoreval/` 提供 CLI 和因子评价流程，`factors/` 保存本轮单因子清单，`reports/` 是本地运行生成的研究结果（不会提交大体积行情数据）。
