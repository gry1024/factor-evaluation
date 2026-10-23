import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hac_tstat(values, max_lag=20):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    centered = x - x.mean()
    lrv = np.dot(centered, centered) / n
    lag = min(max_lag, n - 1)
    for k in range(1, lag + 1):
        gamma = np.dot(centered[k:], centered[:-k]) / n
        lrv += 2 * (1 - k / (lag + 1)) * gamma
    se = np.sqrt(max(lrv, 0) / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def make_dashboard(out, name, ric, quantile, daily):
    dates = pd.to_datetime(daily.index)
    strategy_nav = daily[("Single factor", "nav")]
    benchmark_nav = daily[("CSI 300 index", "nav")]
    drawdown = daily[("Single factor", "drawdown")]
    returns = daily[("Single factor", "return")]
    monthly = (1 + returns).resample("ME").prod() - 1
    rolling_ric = ric.rolling(63, min_periods=30).mean()
    qmeans = quantile[[f"Q{i}" for i in range(1, 6)]].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(dates, strategy_nav, label="Factor portfolio", color="#1565c0", lw=1.8)
    ax.plot(dates, benchmark_nav, label="CSI 300", color="#777777", lw=1.4)
    ax.set_title("Net asset value (after trading costs)")
    ax.set_ylabel("Growth of 1 CNY")
    ax.grid(alpha=.25); ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.fill_between(dates, drawdown.to_numpy() * 100, 0, color="#ef6c00", alpha=.45)
    ax.plot(dates, drawdown.to_numpy() * 100, color="#c62828", lw=1)
    ax.set_title("Portfolio drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    ax.plot(pd.to_datetime(ric.index), ric, color="#90a4ae", lw=.7, alpha=.7, label="Daily RankIC")
    ax.plot(pd.to_datetime(rolling_ric.index), rolling_ric, color="#6a1b9a", lw=1.8, label="63-session mean")
    ax.axhline(0, color="#444444", lw=.8)
    ax.set_title("RankIC stability")
    ax.set_ylabel("RankIC")
    ax.grid(alpha=.25); ax.legend(frameon=False)

    ax = axes[1, 1]
    qmeans.mul(100).plot(kind="bar", ax=ax, color=["#90caf9", "#64b5f6", "#42a5f5", "#1e88e5", "#0d47a1"])
    ax.set_title("Mean forward return by factor quintile")
    ax.set_ylabel("Next 20-session open-to-open return (%)")
    ax.set_xlabel("Low factor score (Q1) → high factor score (Q5)")
    ax.tick_params(axis="x", rotation=0); ax.grid(axis="y", alpha=.25)
    fig.suptitle(f"{out.name} | {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}", fontsize=15)
    fig.savefig(out / "dashboard.png", dpi=160)
    plt.close(fig)

    monthly.index = monthly.index.to_period("M")
    monthly_frame = monthly.to_frame("return")
    monthly_frame.index.name = "month"
    monthly_frame.to_csv(out / "monthly_returns.csv", float_format="%.8f")
    heat = pd.DataFrame(index=sorted(set(monthly.index.year)), columns=range(1, 13), dtype=float)
    for month, value in monthly.items():
        heat.loc[month.year, month.month] = value * 100
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=-15, vmax=15)
    ax.set_yticks(range(len(heat.index)), labels=heat.index)
    ax.set_xticks(range(12), labels=[f"{m:02d}" for m in range(1, 13)])
    ax.set_xlabel("Month"); ax.set_title(f"{out.name} | Monthly portfolio returns (%)")
    for i in range(len(heat.index)):
        for j in range(12):
            value = heat.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Return (%)")
    fig.savefig(out / "monthly_heatmap.png", dpi=160)
    plt.close(fig)


def evaluate_one(root: Path, item: dict, out: Path, reuse_existing=False):
    sys.path.insert(0, str(root))
    from quantbt.config import load_pool
    from quantbt.data import MarketData
    from quantbt.expressions import Expression
    from quantbt.metrics import daily_corr, normalize
    from quantbt.runner import run

    pool_path = root / item["pool"]
    pool, config = load_pool(pool_path)
    data = MarketData(root / "feathers", config.strict_features)
    days = data.eval_indices
    signal_days = days - 1
    eligible = data.eligible(config.min_listing_days)
    labels = data.forward_returns(config.label_days)
    labels[~eligible] = np.nan
    expr = Expression(pool["exprs"][0])
    scores = normalize(expr.evaluate(data), eligible)
    ic = daily_corr(scores[signal_days], labels[signal_days])
    ric = daily_corr(scores[signal_days], labels[signal_days], ranked=True)

    # Daily rank buckets with deterministic tie handling; returns are overlapping
    # forward labels and are kept as diagnostics instead of annualized performance.
    group_daily = np.full((len(signal_days), 5), np.nan)
    for j, day in enumerate(signal_days):
        valid = np.isfinite(scores[day]) & np.isfinite(labels[day])
        count = int(valid.sum())
        if count < 15:
            continue
        ranks = pd.Series(scores[day, valid]).rank(method="first", pct=True).to_numpy()
        bucket = np.minimum((ranks * 5).astype(int), 4)
        y = labels[day, valid]
        for q in range(5):
            members = bucket == q
            if members.any():
                group_daily[j, q] = float(y[members].mean())
    dates = data.dates[signal_days]
    quantile = pd.DataFrame(group_daily, index=dates, columns=[f"Q{i}" for i in range(1, 6)])
    quantile["Q5_minus_Q1"] = quantile.Q5 - quantile.Q1
    quantile.index.name = "signal_date"
    out.mkdir(parents=True, exist_ok=True)
    quantile.to_csv(out / "quantile_returns.csv", float_format="%.8f")
    pd.DataFrame({"IC": ic, "RankIC": ric}, index=dates).rename_axis("signal_date").to_csv(out / "daily_ic.csv", float_format="%.8f")

    backtest_dir = out / "backtest"
    if not (reuse_existing and (backtest_dir / "metrics.json").is_file() and (backtest_dir / "daily.csv").is_file()):
        run(pool_path, root / "feathers", root / "benchmarks/csi300.csv", backtest_dir)
    metrics = json.loads((backtest_dir / "metrics.json").read_text(encoding="utf-8"))["Single factor"]
    daily = pd.read_csv(backtest_dir / "daily.csv", header=[0, 1], index_col=0, parse_dates=True)
    portfolio_returns = daily[("Single factor", "return")].to_numpy(dtype=float)
    benchmark_returns = daily[("CSI 300 index", "return")].to_numpy(dtype=float)
    downside = np.minimum(portfolio_returns, 0)
    var95 = float(np.quantile(portfolio_returns, .05))
    active = portfolio_returns - benchmark_returns
    beta = float(np.cov(portfolio_returns, benchmark_returns, ddof=1)[0, 1] / np.var(benchmark_returns, ddof=1))
    annual_returns = ((1 + daily.loc[:, (slice(None), "return")]).groupby(daily.index.year).prod() - 1)
    annual_returns.index.name = "year"
    annual_returns.to_csv(out / "annual_returns.csv", float_format="%.8f")
    make_dashboard(out, item["name"], pd.Series(ric, index=dates), quantile, daily)
    metrics.update({
        "factor_id": item["id"], "factor_name": item["name"],
        "ic_mean": float(np.nanmean(ic)), "icir": float(np.nanmean(ic) / np.nanstd(ic, ddof=1)),
        "rank_ic_mean": float(np.nanmean(ric)), "rank_icir": float(np.nanmean(ric) / np.nanstd(ric, ddof=1)),
        "ic_hac_tstat_lag20": hac_tstat(ic), "rank_ic_hac_tstat_lag20": hac_tstat(ric),
        "ic_positive_rate": float(np.nanmean(ic > 0)), "rank_ic_positive_rate": float(np.nanmean(ric > 0)),
        "label_days": config.label_days, "quantile_spread_mean": float(quantile.Q5_minus_Q1.mean()),
        "quantile_monotonic_days": int((quantile[[f"Q{i}" for i in range(1, 6)]].diff(axis=1).iloc[:, 1:] > 0).all(axis=1).sum()),
        "sortino": float(np.mean(portfolio_returns) / np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)) if np.any(downside < 0) else float("nan"),
        "return_skewness": float(pd.Series(portfolio_returns).skew()),
        "return_excess_kurtosis": float(pd.Series(portfolio_returns).kurt()),
        "historical_var_95_daily": var95,
        "historical_cvar_95_daily": float(portfolio_returns[portfolio_returns <= var95].mean()),
        "beta_vs_csi300": beta,
        "annualized_alpha_vs_csi300": float((np.mean(portfolio_returns - beta * benchmark_returns)) * 252),
        "downside_deviation_annualized": float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)),
        "quantile_q1_mean_forward_return": float(quantile.Q1.mean()),
        "quantile_q2_mean_forward_return": float(quantile.Q2.mean()),
        "quantile_q3_mean_forward_return": float(quantile.Q3.mean()),
        "quantile_q4_mean_forward_return": float(quantile.Q4.mean()),
        "quantile_q5_mean_forward_return": float(quantile.Q5.mean()),
        "quantile_top_bottom_spread_std": float(quantile.Q5_minus_Q1.std()),
        "quantile_top_bottom_spread_positive_rate": float((quantile.Q5_minus_Q1 > 0).mean()),
        "average_daily_active_return": float(np.mean(active)),
    })
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate selected single factors using the quantbt backtest engine")
    parser.add_argument("--backtest-root", type=Path, required=True, help="Path to quant-backtest-system")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--factors", type=Path, default=Path(__file__).resolve().parent.parent / "factors/selected.json")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing backtest outputs in each factor directory")
    args = parser.parse_args()
    root, output = args.backtest_root.resolve(), args.output.resolve()
    items = json.loads(args.factors.read_text(encoding="utf-8"))
    results = [evaluate_one(root, item, output / item["id"], args.reuse_existing) for item in items]
    frame = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    frame.to_csv(output / "summary.csv", index=False, float_format="%.6f")
    cols = ["factor_name", "ic_mean", "icir", "rank_ic_mean", "rank_icir", "annual_return", "max_drawdown", "sharpe", "sortino", "calmar", "information_ratio", "quantile_spread_mean", "avg_turnover"]
    labels = {"factor_name":"因子", "ic_mean":"IC", "icir":"ICIR", "rank_ic_mean":"RIC", "rank_icir":"RICIR", "annual_return":"ARR(年化收益)", "max_drawdown":"MDD", "sharpe":"SR", "sortino":"Sortino", "calmar":"Calmar", "information_ratio":"IR", "quantile_spread_mean":"Q5-Q1未来20日", "avg_turnover":"平均换手"}
    display = frame[cols].rename(columns=labels)
    table = "| " + " | ".join(display.columns) + " |\n| " + " | ".join(["---"] * len(display.columns)) + " |\n"
    percentage_columns = {"ARR(年化收益)", "MDD", "Q5-Q1未来20日", "平均换手"}
    for row in display.itertuples(index=False, name=None):
        formatted = []
        for column, value in zip(display.columns, row):
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.2%}" if column in percentage_columns else f"{value:.4f}")
            else:
                formatted.append(str(value))
        table += "| " + " | ".join(formatted) + " |\n"
    annual = []
    for result in results:
        frame_annual = pd.read_csv(output / result["factor_id"] / "annual_returns.csv", skiprows=2, index_col=0)
        annual.append((result["factor_name"], frame_annual.iloc[:, 0]))
    years = sorted(set().union(*(series.index for _, series in annual)))
    annual_table = "| 年份 | " + " | ".join(name for name, _ in annual) + " |\n| --- | " + " | ".join(["---"] * len(annual)) + " |\n"
    for year in years:
        annual_table += f"| {year} | " + " | ".join(f"{series.get(year, np.nan):.2%}" for _, series in annual) + " |\n"
    details = []
    for result in results:
        key_labels = {
            "total_return":"累计收益", "annual_volatility":"年化波动率", "sortino":"Sortino 比率", "calmar":"Calmar 比率", "information_ratio":"信息比率", "excess_total_return":"相对基准超额收益", "daily_win_rate":"日胜率", "ic_mean":"IC均值", "icir":"ICIR", "ic_hac_tstat_lag20":"IC HAC t统计", "rank_ic_hac_tstat_lag20":"RankIC HAC t统计", "ic_positive_rate":"IC正值比例", "rank_ic_positive_rate":"RankIC正值比例", "quantile_q1_mean_forward_return":"Q1未来20日收益", "quantile_q2_mean_forward_return":"Q2未来20日收益", "quantile_q3_mean_forward_return":"Q3未来20日收益", "quantile_q4_mean_forward_return":"Q4未来20日收益", "quantile_q5_mean_forward_return":"Q5未来20日收益", "quantile_top_bottom_spread_positive_rate":"Q5-Q1正收益比例", "beta_vs_csi300":"沪深300 Beta", "annualized_alpha_vs_csi300":"年化Alpha估计", "historical_var_95_daily":"日VaR 95%", "historical_cvar_95_daily":"日CVaR 95%", "return_skewness":"日收益偏度", "return_excess_kurtosis":"日收益超额峰度", "avg_exposure":"平均仓位", "total_commission":"累计佣金", "total_stamp_duty":"累计印花税"
        }
        detail = "| 指标 | 数值 |\n| --- | ---: |\n"
        for key, label in key_labels.items():
            value = result.get(key, float("nan"))
            formatted = f"{value:.4%}" if key in {"total_return", "annual_volatility", "excess_total_return", "daily_win_rate", "ic_positive_rate", "rank_ic_positive_rate", "quantile_q1_mean_forward_return", "quantile_q2_mean_forward_return", "quantile_q3_mean_forward_return", "quantile_q4_mean_forward_return", "quantile_q5_mean_forward_return", "quantile_top_bottom_spread_positive_rate"} else (f"{value:,.2f}" if key in {"total_commission", "total_stamp_duty"} else f"{value:.4f}")
            detail += f"| {label} | {formatted} |\n"
        details.append(f"### {result['factor_name']}\n\n#### 组合与预测指标\n\n{detail}\n#### 净值、回撤、IC 与分层表现\n\n![{result['factor_name']}评价图]({result['factor_id']}/dashboard.png)\n\n#### 月收益热力图\n\n![{result['factor_name']}月收益热力图]({result['factor_id']}/monthly_heatmap.png)\n")
    (output / "summary.md").write_text("# 单因子评价结果\n\n" + table + "\n## 年度收益\n\n" + annual_table + "\n## 因子诊断\n\n" + "\n".join(details) + "\n分层收益为重叠的未来 20 日开盘到开盘诊断值，不用于 Sharpe 年化；Sharpe 等组合指标来自逐日真实撮合回测，含手续费。HAC t 统计采用最多 20 阶 Newey-West 修正。VaR/CVaR 使用历史日收益分位数。\n", encoding="utf-8")
    print(f"Wrote {output / 'summary.md'}")


if __name__ == "__main__":
    main()
