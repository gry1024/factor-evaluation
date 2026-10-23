import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_one(root: Path, item: dict, out: Path):
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
    run(pool_path, root / "feathers", root / "benchmarks/csi300.csv", backtest_dir)
    metrics = json.loads((backtest_dir / "metrics.json").read_text(encoding="utf-8"))["Single factor"]
    metrics.update({
        "factor_id": item["id"], "factor_name": item["name"],
        "ic_mean": float(np.nanmean(ic)), "icir": float(np.nanmean(ic) / np.nanstd(ic, ddof=1)),
        "rank_ic_mean": float(np.nanmean(ric)), "rank_icir": float(np.nanmean(ric) / np.nanstd(ric, ddof=1)),
        "ic_positive_rate": float(np.nanmean(ic > 0)), "rank_ic_positive_rate": float(np.nanmean(ric > 0)),
        "label_days": config.label_days, "quantile_spread_mean": float(quantile.Q5_minus_Q1.mean()),
        "quantile_monotonic_days": int((quantile[[f"Q{i}" for i in range(1, 6)]].diff(axis=1).iloc[:, 1:] > 0).all(axis=1).sum()),
    })
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate selected single factors using the quantbt backtest engine")
    parser.add_argument("--backtest-root", type=Path, required=True, help="Path to quant-backtest-system")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--factors", type=Path, default=Path(__file__).resolve().parent.parent / "factors/selected.json")
    args = parser.parse_args()
    root, output = args.backtest_root.resolve(), args.output.resolve()
    items = json.loads(args.factors.read_text(encoding="utf-8"))
    results = [evaluate_one(root, item, output / item["id"]) for item in items]
    frame = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    frame.to_csv(output / "summary.csv", index=False, float_format="%.6f")
    cols = ["factor_name", "ic_mean", "rank_ic_mean", "rank_icir", "quantile_spread_mean", "annual_return", "sharpe", "max_drawdown", "information_ratio", "avg_turnover"]
    labels = {"factor_name":"因子", "ic_mean":"IC", "rank_ic_mean":"RankIC", "rank_icir":"RankICIR", "quantile_spread_mean":"Q5-Q1(20日)", "annual_return":"年化收益", "sharpe":"Sharpe", "max_drawdown":"最大回撤", "information_ratio":"信息比率", "avg_turnover":"平均换手"}
    display = frame[cols].rename(columns=labels)
    table = "| " + " | ".join(display.columns) + " |\n| " + " | ".join(["---"] * len(display.columns)) + " |\n"
    for row in display.itertuples(index=False, name=None):
        table += "| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |\n"
    (output / "summary.md").write_text("# 单因子评价结果\n\n" + table + "\n分层收益为重叠的未来 20 日开盘到开盘诊断值，不用于 Sharpe 年化；Sharpe 等组合指标来自逐日真实撮合回测，含手续费。\n", encoding="utf-8")
    print(f"Wrote {output / 'summary.md'}")


if __name__ == "__main__":
    main()
