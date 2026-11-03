#!/usr/bin/env python3
"""
QuantSearch Visualization & Analytics Suite.
Computes and plots:
- Portfolio value trajectories ("the graphic")
- Daily returns histogram
- Ordered returns (empirical distribution)
- Logarithm of daily returns (time series & cumulative)
- Sharpe ratio comparison
- Scatter plot of return vs standard deviation (risk-return profile)
- Maximum drawdown (MDD) underwater curves
- Number of orders & portfolio turnover activity
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def find_directories():
    current = Path.cwd()
    for base in [current, current.parent, current.parent.parent]:
        results_dir = base / "_results"
        plots_dir = base / "_plots"
        if results_dir.is_dir():
            plots_dir.mkdir(parents=True, exist_ok=True)
            return results_dir, plots_dir
    raise RuntimeError("Could not find '_results' directory.")


def resolve_run_file(results_dir: Path, run_arg: str | None = None) -> tuple[str, Path]:
    if run_arg:
        arg_path = Path(run_arg)
        if arg_path.is_file():
            run_id = arg_path.stem
            return run_id, arg_path
        candidate = results_dir / f"{run_arg}.csv"
        if candidate.is_file():
            return run_arg, candidate
        candidate_direct = results_dir / run_arg
        if candidate_direct.is_file():
            return candidate_direct.stem, candidate_direct
        raise FileNotFoundError(f"Specified run CSV file not found: {run_arg}")

    csv_files = [p for p in results_dir.glob("*.csv") if p.stat().st_size > 0]
    if not csv_files:
        raise FileNotFoundError(f"No non-empty CSV files found in {results_dir}")
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest_file.stem, latest_file


def format_agent_label(name: str) -> str:
    aliases = {
        "my": "Convolutional Agent",
        "baseline": "Equal Weight Baseline",
        "price_matrix": "Price Matrix (MLP)",
        "volume_matrix": "Volume Matrix (MLP)",
        "dual_matrix": "Dual Matrix (MLP)",
        "convolutional": "Convolutional (Dual)",
        "convolutional_both": "Convolutional (Dual)",
        "convolutional_dual": "Convolutional (Dual)",
        "convolutional_price": "Convolutional (Price Only)",
        "convolutional_close": "Convolutional (Price Only)",
        "convolutional_volume": "Convolutional (Volume Only)",
        "same_percent": "Same Percent (Baseline)",
        "best_performing": "Best Performing (Baseline)",
    }
    return aliases.get(name, name.replace("_", " ").title())


def compute_agent_metrics(df: pd.DataFrame, agent_names: list[str]) -> dict[str, dict]:
    metrics = {}
    trading_days_per_year = 252

    for agent in agent_names:
        money_col = f"{agent}_money"
        series = df[money_col].to_numpy(dtype=float)
        dates = df["date"]

        # Simple and log returns
        prev_series = series[:-1]
        curr_series = series[1:]

        # Handle zero or non-positive money values safely
        valid_mask = (prev_series > 0) & (curr_series > 0)
        daily_returns = np.zeros(len(series))
        daily_log_returns = np.zeros(len(series))

        if len(series) > 1:
            raw_returns = np.divide(
                curr_series - prev_series,
                prev_series,
                out=np.zeros_like(curr_series),
                where=prev_series > 0,
            )
            daily_returns[1:] = raw_returns

            raw_log_returns = np.zeros_like(curr_series)
            raw_log_returns[valid_mask] = np.log(curr_series[valid_mask] / prev_series[valid_mask])
            daily_log_returns[1:] = raw_log_returns

        cum_log_returns = np.cumsum(daily_log_returns)

        # Performance summary metrics
        total_days = max(1, len(series) - 1)
        initial_money = series[0] if len(series) > 0 else 10000.0
        final_money = series[-1] if len(series) > 0 else initial_money
        total_return = (final_money - initial_money) / initial_money if initial_money > 0 else 0.0

        ret_slice = daily_returns[1:] if len(daily_returns) > 1 else np.array([0.0])
        mean_daily_return = np.mean(ret_slice)
        std_daily_return = np.std(ret_slice, ddof=1) if len(ret_slice) > 1 else 0.0

        ann_return = mean_daily_return * trading_days_per_year
        ann_volatility = std_daily_return * np.sqrt(trading_days_per_year)
        sharpe_ratio = (
            (mean_daily_return / std_daily_return) * np.sqrt(trading_days_per_year)
            if std_daily_return > 1e-12
            else 0.0
        )

        # Drawdown calculation
        running_peak = np.maximum.accumulate(series)
        drawdown = np.divide(
            series - running_peak,
            running_peak,
            out=np.zeros_like(series),
            where=running_peak > 0,
        )
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        # Orders and portfolio turnover
        stock_cols = [c for c in df.columns if c.startswith(f"{agent}_percent_stock_")]
        if not stock_cols and agent == "my":
            stock_cols = [c for c in df.columns if c.startswith("percent_stock_")]

        daily_turnover = np.zeros(len(df))
        daily_orders = np.zeros(len(df))

        if stock_cols:
            weights = df[stock_cols].to_numpy(dtype=float) / 100.0  # convert % to fraction
            if len(weights) > 1:
                weight_diffs = np.abs(np.diff(weights, axis=0))
                # Turnover is half of sum of absolute weight changes
                daily_turnover[1:] = 0.5 * np.sum(weight_diffs, axis=1)
                # Count orders as weight changes greater than 0.1%
                daily_orders[1:] = np.sum(weight_diffs > 0.001, axis=1)

        total_orders = int(np.sum(daily_orders))
        mean_daily_orders = float(np.mean(daily_orders[1:])) if len(daily_orders) > 1 else 0.0
        mean_turnover = float(np.mean(daily_turnover[1:])) if len(daily_turnover) > 1 else 0.0

        metrics[agent] = {
            "initial_money": initial_money,
            "final_money": final_money,
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_volatility": ann_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "daily_returns": daily_returns,
            "daily_log_returns": daily_log_returns,
            "cum_log_returns": cum_log_returns,
            "drawdown": drawdown,
            "daily_turnover": daily_turnover,
            "daily_orders": daily_orders,
            "total_orders": total_orders,
            "mean_daily_orders": mean_daily_orders,
            "mean_turnover": mean_turnover,
        }

    return metrics


def print_performance_table(run_id: str, agent_names: list[str], metrics: dict[str, dict]):
    print("\n" + "=" * 105)
    print(f" QUANTSEARCH EXPERIMENT REPORT: {run_id}")
    print("=" * 105)
    header = (
        f"{'Agent':<28} | {'Final ($)':>11} | {'Tot Ret (%)':>11} | {'Ann Ret (%)':>11} | "
        f"{'Ann Vol (%)':>11} | {'Sharpe':>8} | {'Max DD (%)':>10} | {'Orders':>7} | {'Turnover (%)':>12}"
    )
    print(header)
    print("-" * 105)
    for agent in agent_names:
        m = metrics[agent]
        label = format_agent_label(agent)
        print(
            f"{label:<28} | "
            f"{m['final_money']:>11.2f} | "
            f"{m['total_return'] * 100:>10.2f}% | "
            f"{m['ann_return'] * 100:>10.2f}% | "
            f"{m['ann_volatility'] * 100:>10.2f}% | "
            f"{m['sharpe_ratio']:>8.2f} | "
            f"{m['max_drawdown'] * 100:>9.2f}% | "
            f"{m['total_orders']:>7d} | "
            f"{m['mean_turnover'] * 100:>11.2f}%"
        )
    print("=" * 105 + "\n")


def generate_plots(
    run_id: str,
    df: pd.DataFrame,
    agent_names: list[str],
    metrics: dict[str, dict],
    plots_dir: Path,
):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = plt.cm.tab10.colors
    agent_colors = {agent: colors[idx % len(colors)] for idx, agent in enumerate(agent_names)}

    dates = df["date"]
    years = dates.dt.year.unique()

    # -------------------------------------------------------------
    # 1. Master Comprehensive Dashboard (8 Subplots)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle(f"QuantSearch Evaluation Dashboard - Run {run_id}", fontsize=18, fontweight="bold", y=0.995)

    # (A) Portfolio Value Trajectory
    ax_money = axes[0, 0]
    for agent in agent_names:
        ax_money.plot(dates, df[f"{agent}_money"], label=format_agent_label(agent), color=agent_colors[agent], linewidth=1.8)
    ax_money.axhline(y=10_000, color="black", linestyle="--", linewidth=1.2, label="Starting Cash ($10,000)")
    for year in years:
        ax_money.axvline(pd.Timestamp(f"{year}-01-01"), color="lightgray", linestyle=":", linewidth=0.8)
    ax_money.set_title("Portfolio Money Trajectory Over Time", fontsize=13, fontweight="bold")
    ax_money.set_ylabel("Portfolio Value ($)")
    ax_money.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax_money.legend(loc="upper left", framealpha=0.9)

    # (B) Maximum Drawdown Underwater Plot
    ax_dd = axes[0, 1]
    for agent in agent_names:
        dd = metrics[agent]["drawdown"] * 100.0
        ax_dd.plot(dates, dd, label=f"{format_agent_label(agent)} (Max: {metrics[agent]['max_drawdown']*100:.1f}%)",
                   color=agent_colors[agent], linewidth=1.5)
        ax_dd.fill_between(dates, dd, 0, color=agent_colors[agent], alpha=0.1)
    for year in years:
        ax_dd.axvline(pd.Timestamp(f"{year}-01-01"), color="lightgray", linestyle=":", linewidth=0.8)
    ax_dd.set_title("Drawdown Underwater Profile (%)", fontsize=13, fontweight="bold")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax_dd.legend(loc="lower left", framealpha=0.9)

    # (C) Daily Returns Histogram
    ax_hist = axes[1, 0]
    for agent in agent_names:
        rets = metrics[agent]["daily_returns"][1:] * 100.0
        # Clip display range for extreme outliers to keep histogram informative
        q_low, q_high = np.percentile(rets, [0.5, 99.5])
        filtered = rets[(rets >= q_low) & (rets <= q_high)]
        ax_hist.hist(filtered, bins=45, alpha=0.45, density=True, label=format_agent_label(agent), color=agent_colors[agent])
    ax_hist.axvline(x=0.0, color="black", linestyle="--", linewidth=1)
    ax_hist.set_title("Daily Returns Distribution (Histogram)", fontsize=13, fontweight="bold")
    ax_hist.set_xlabel("Daily Return (%)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right", framealpha=0.9)

    # (D) Ordered Returns Plot
    ax_ord = axes[1, 1]
    for agent in agent_names:
        sorted_rets = np.sort(metrics[agent]["daily_returns"][1:] * 100.0)
        ranks = np.linspace(0, 100, len(sorted_rets))
        ax_ord.plot(ranks, sorted_rets, label=format_agent_label(agent), color=agent_colors[agent], linewidth=1.8)
    ax_ord.axhline(y=0.0, color="black", linestyle="--", linewidth=1)
    ax_ord.set_title("Ordered Daily Returns (Empirical Quantiles)", fontsize=13, fontweight="bold")
    ax_ord.set_xlabel("Percentile (%)")
    ax_ord.set_ylabel("Ordered Return (%)")
    ax_ord.legend(loc="upper left", framealpha=0.9)

    # (E) Logarithm of Daily Returns Time Series
    ax_log = axes[2, 0]
    for agent in agent_names:
        ax_log.plot(dates, metrics[agent]["cum_log_returns"], label=f"{format_agent_label(agent)} (Cum Log Ret)",
                    color=agent_colors[agent], linewidth=1.6)
    for year in years:
        ax_log.axvline(pd.Timestamp(f"{year}-01-01"), color="lightgray", linestyle=":", linewidth=0.8)
    ax_log.axhline(y=0.0, color="black", linestyle="--", linewidth=0.8)
    ax_log.set_title("Cumulative Logarithmic Returns Over Time", fontsize=13, fontweight="bold")
    ax_log.set_ylabel("ln(V_t / V_0)")
    ax_log.legend(loc="upper left", framealpha=0.9)

    # (F) Scatter Plot: Return vs Standard Deviation (Risk-Reward)
    ax_scatter = axes[2, 1]
    for agent in agent_names:
        vol = metrics[agent]["ann_volatility"] * 100.0
        ret = metrics[agent]["ann_return"] * 100.0
        sr = metrics[agent]["sharpe_ratio"]
        ax_scatter.scatter(vol, ret, color=agent_colors[agent], s=160, zorder=5, label=f"{format_agent_label(agent)} (SR: {sr:.2f})")
        ax_scatter.annotate(
            format_agent_label(agent),
            (vol, ret),
            textcoords="offset points",
            xytext=(10, 5),
            fontweight="semibold",
            fontsize=10,
        )
    ax_scatter.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8)
    ax_scatter.set_title("Risk-Return Profile (Annualized Volatility vs Return)", fontsize=13, fontweight="bold")
    ax_scatter.set_xlabel("Annualized Volatility (%)")
    ax_scatter.set_ylabel("Annualized Return (%)")
    ax_scatter.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax_scatter.xaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax_scatter.legend(loc="best", framealpha=0.9)

    # (G) Sharpe Ratio Comparison Bar Chart
    ax_sr = axes[3, 0]
    agent_labels = [format_agent_label(a) for a in agent_names]
    sr_values = [metrics[a]["sharpe_ratio"] for a in agent_names]
    bar_colors = [agent_colors[a] for a in agent_names]
    bars = ax_sr.bar(agent_labels, sr_values, color=bar_colors, alpha=0.85, edgecolor="black", width=0.55)
    ax_sr.axhline(y=0.0, color="black", linestyle="-", linewidth=0.8)
    for bar, val in zip(bars, sr_values):
        y_pos = val + (0.05 if val >= 0 else -0.15)
        ax_sr.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontweight="bold")
    ax_sr.set_title("Annualized Sharpe Ratio Comparison", fontsize=13, fontweight="bold")
    ax_sr.set_ylabel("Sharpe Ratio")
    ax_sr.tick_params(axis="x", rotation=15)

    # (H) Portfolio Rebalancing Turnover / Orders
    ax_orders = axes[3, 1]
    has_order_data = any(metrics[a]["total_orders"] > 0 for a in agent_names)
    if has_order_data:
        for agent in agent_names:
            cum_turnover = np.cumsum(metrics[agent]["daily_turnover"]) * 100.0
            ax_orders.plot(dates, cum_turnover, label=f"{format_agent_label(agent)} (Orders: {metrics[agent]['total_orders']})",
                           color=agent_colors[agent], linewidth=1.7)
        ax_orders.set_title("Cumulative Portfolio Turnover / Order Intensity (%)", fontsize=13, fontweight="bold")
        ax_orders.set_ylabel("Cumulative Turnover (%)")
    else:
        # Fallback to daily log return volatility if weights aren't recorded
        for agent in agent_names:
            ax_orders.plot(dates, metrics[agent]["daily_log_returns"], label=format_agent_label(agent),
                           color=agent_colors[agent], alpha=0.6, linewidth=1)
        ax_orders.set_title("Daily Log Returns Trace", fontsize=13, fontweight="bold")
        ax_orders.set_ylabel("Daily Log Return")
    for year in years:
        ax_orders.axvline(pd.Timestamp(f"{year}-01-01"), color="lightgray", linestyle=":", linewidth=0.8)
    ax_orders.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    dashboard_path = plots_dir / f"{run_id}.png"
    dashboard_alt_path = plots_dir / f"{run_id}_dashboard.png"
    plt.savefig(dashboard_path, dpi=200)
    plt.savefig(dashboard_alt_path, dpi=200)
    plt.close(fig)
    print(f" Saved master dashboard to: {dashboard_path}")

    # -------------------------------------------------------------
    # 2. Individual Focused High-Res Plots
    # -------------------------------------------------------------
    # (1) Portfolio Value Only
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    for agent in agent_names:
        ax1.plot(dates, df[f"{agent}_money"], label=format_agent_label(agent), color=agent_colors[agent], linewidth=2.0)
    ax1.axhline(y=10_000, color="black", linestyle="--", linewidth=1.2, label="Starting Cash ($10,000)")
    for year in years:
        ax1.axvline(pd.Timestamp(f"{year}-01-01"), color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title(f"Portfolio Value Comparison ({run_id})", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax1.legend(loc="upper left")
    fig1.tight_layout()
    fig1.savefig(plots_dir / f"{run_id}_portfolio_value.png", dpi=200)
    plt.close(fig1)

    # (2) Returns Histogram Only
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for agent in agent_names:
        rets = metrics[agent]["daily_returns"][1:] * 100.0
        q_low, q_high = np.percentile(rets, [0.5, 99.5])
        filtered = rets[(rets >= q_low) & (rets <= q_high)]
        ax2.hist(filtered, bins=50, alpha=0.5, density=True, label=format_agent_label(agent), color=agent_colors[agent])
    ax2.axvline(x=0.0, color="black", linestyle="--", linewidth=1)
    ax2.set_title(f"Histogram of Daily Returns ({run_id})", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Daily Return (%)")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(plots_dir / f"{run_id}_returns_histogram.png", dpi=200)
    plt.close(fig2)

    # (3) Ordered Returns Only
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for agent in agent_names:
        sorted_rets = np.sort(metrics[agent]["daily_returns"][1:] * 100.0)
        ranks = np.linspace(0, 100, len(sorted_rets))
        ax3.plot(ranks, sorted_rets, label=format_agent_label(agent), color=agent_colors[agent], linewidth=2.0)
    ax3.axhline(y=0.0, color="black", linestyle="--", linewidth=1)
    ax3.set_title(f"Ordered Daily Returns ({run_id})", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Percentile (%)")
    ax3.set_ylabel("Return (%)")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(plots_dir / f"{run_id}_ordered_returns.png", dpi=200)
    plt.close(fig3)

    # (4) Log Returns Only
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    for agent in agent_names:
        ax4.plot(dates, metrics[agent]["cum_log_returns"], label=format_agent_label(agent), color=agent_colors[agent], linewidth=1.8)
    for year in years:
        ax4.axvline(pd.Timestamp(f"{year}-01-01"), color="gray", linestyle="--", linewidth=0.8)
    ax4.axhline(y=0.0, color="black", linestyle="--", linewidth=0.8)
    ax4.set_title(f"Cumulative Log Returns: ln(V_t / V_0) ({run_id})", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Log Return")
    ax4.legend(loc="upper left")
    fig4.tight_layout()
    fig4.savefig(plots_dir / f"{run_id}_log_returns.png", dpi=200)
    plt.close(fig4)

    # (5) Scatter Risk vs Return
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    for agent in agent_names:
        vol = metrics[agent]["ann_volatility"] * 100.0
        ret = metrics[agent]["ann_return"] * 100.0
        sr = metrics[agent]["sharpe_ratio"]
        ax5.scatter(vol, ret, color=agent_colors[agent], s=200, zorder=5, label=f"{format_agent_label(agent)} (SR: {sr:.2f})")
        ax5.annotate(format_agent_label(agent), (vol, ret), textcoords="offset points", xytext=(10, 5), fontweight="bold")
    ax5.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8)
    ax5.set_title(f"Risk vs Return Profile ({run_id})", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Annualized Standard Deviation (Volatility) (%)")
    ax5.set_ylabel("Annualized Return (%)")
    ax5.legend(loc="best")
    fig5.tight_layout()
    fig5.savefig(plots_dir / f"{run_id}_risk_return_scatter.png", dpi=200)
    plt.close(fig5)

    # (6) Drawdown Underwater Plot
    fig6, ax6 = plt.subplots(figsize=(14, 6))
    for agent in agent_names:
        dd = metrics[agent]["drawdown"] * 100.0
        ax6.plot(dates, dd, label=f"{format_agent_label(agent)} (Max DD: {metrics[agent]['max_drawdown']*100:.2f}%)",
                 color=agent_colors[agent], linewidth=1.8)
        ax6.fill_between(dates, dd, 0, color=agent_colors[agent], alpha=0.1)
    for year in years:
        ax6.axvline(pd.Timestamp(f"{year}-01-01"), color="gray", linestyle="--", linewidth=0.8)
    ax6.set_title(f"Maximum Drawdown Underwater Profile ({run_id})", fontsize=14, fontweight="bold")
    ax6.set_xlabel("Date")
    ax6.set_ylabel("Drawdown (%)")
    ax6.legend(loc="lower left")
    fig6.tight_layout()
    fig6.savefig(plots_dir / f"{run_id}_drawdown.png", dpi=200)
    plt.close(fig6)

    # (7) Orders & Turnover Plot
    fig7, ax7 = plt.subplots(figsize=(14, 6))
    if has_order_data:
        for agent in agent_names:
            cum_turnover = np.cumsum(metrics[agent]["daily_turnover"]) * 100.0
            ax7.plot(dates, cum_turnover, label=f"{format_agent_label(agent)} (Total Orders: {metrics[agent]['total_orders']})",
                     color=agent_colors[agent], linewidth=1.8)
        ax7.set_title(f"Cumulative Portfolio Turnover & Order Intensity ({run_id})", fontsize=14, fontweight="bold")
        ax7.set_ylabel("Cumulative Turnover (%)")
    else:
        for agent in agent_names:
            ax7.plot(dates, metrics[agent]["daily_returns"] * 100.0, label=format_agent_label(agent),
                     color=agent_colors[agent], alpha=0.6, linewidth=1)
        ax7.set_title(f"Daily Simple Returns Trace ({run_id})", fontsize=14, fontweight="bold")
        ax7.set_ylabel("Daily Return (%)")
    for year in years:
        ax7.axvline(pd.Timestamp(f"{year}-01-01"), color="gray", linestyle="--", linewidth=0.8)
    ax7.set_xlabel("Date")
    ax7.legend(loc="upper left")
    fig7.tight_layout()
    fig7.savefig(plots_dir / f"{run_id}_orders_turnover.png", dpi=200)
    plt.close(fig7)


def main():
    results_dir, plots_dir = find_directories()
    run_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_id, csv_path = resolve_run_file(results_dir, run_arg)
    print(f"Loading experiment results from: {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"CSV file {csv_path} does not contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Detect agents: either '<name>_money' or legacy 'my_money' / 'baseline_money'
    money_cols = [c for c in df.columns if c.endswith("_money")]
    if not money_cols:
        raise ValueError(f"No money columns found in {csv_path}. Expected columns ending with '_money'.")

    agent_names = [c[:-6] for c in money_cols]
    print(f"Detected agents: {', '.join(agent_names)}")

    metrics = compute_agent_metrics(df, agent_names)
    print_performance_table(run_id, agent_names, metrics)
    generate_plots(run_id, df, agent_names, metrics, plots_dir)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
