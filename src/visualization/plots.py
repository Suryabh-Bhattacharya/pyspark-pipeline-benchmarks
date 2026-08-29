"""Automated Benchmark Plot Generation Module.

Generates Seaborn and Matplotlib visual analytics for PySpark optimization benchmarks:
  1. Dual-axis graphs (Execution Time & Skew Ratio) across all 8 factorial runs (Uniform & Skewed datasets).
  2. Main factor aggregate plots (AQE, Salting, Join Strategy) for average Execution Time, 
     Shuffle Read, and Skew Ratio (Uniform & Skewed datasets).
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style() -> None:
    """Configures global Seaborn and Matplotlib aesthetics."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        }
    )


def _clean_numeric_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts benchmark result dictionaries into a clean, numeric Pandas DataFrame."""
    df = pd.DataFrame(results)

    # Normalize column names for lookups
    col_map = {col: col.strip() for col in df.columns}
    df = df.rename(columns=col_map)

    # Extract numeric Execution Time (Seconds)
    exec_col = [c for c in df.columns if "Exec Time" in c or "duration" in c]
    if exec_col:
        df["Exec_Time_Sec"] = (
            df[exec_col[0]].astype(str).str.replace("s", "", regex=False).astype(float)
        )
    else:
        df["Exec_Time_Sec"] = 0.0

    # Extract numeric Throughput (RPS)
    rps_col = [c for c in df.columns if "Throughput" in c or "rps" in c]
    if rps_col:
        df["Throughput_RPS"] = (
            df[rps_col[0]]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("RPS", "", regex=False)
            .astype(float)
        )
    else:
        df["Throughput_RPS"] = 0.0

    # Extract numeric Skew Ratio
    skew_col = [c for c in df.columns if "Skew" in c or "skew_ratio" in c]
    if skew_col:
        df["Skew_Ratio_Num"] = (
            df[skew_col[0]].astype(str).str.replace("x", "", regex=False).astype(float)
        )
    else:
        df["Skew_Ratio_Num"] = 1.0

    # Extract numeric Shuffle Read (MB)
    shuffle_col = [c for c in df.columns if "Shuffle" in c or "shuffle_mb" in c]
    if shuffle_col:
        df["Shuffle_Read_MB"] = (
            df[shuffle_col[0]]
            .astype(str)
            .str.replace(" MB", "", regex=False)
            .str.replace("MB", "", regex=False)
            .astype(float)
        )
    else:
        df["Shuffle_Read_MB"] = 0.0

    return df


def plot_dual_axis_factorial(
    results: List[Dict[str, Any]],
    dataset_type: str,
    output_dir: Union[str, Path] = "reports/figures",
) -> Path:
    """Plot 1: Dual-axis plot for all 8 factorial runs showing Execution Time and Skew Ratio."""
    setup_plot_style()
    df = _clean_numeric_data(results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"factorial_dual_axis_{dataset_type.lower()}.png"
    file_path = output_path / filename

    # Create composite configuration label
    df["Config_Label"] = (
        "Run " + df["Run"].astype(str) + "\nAQE: " + df["AQE"].astype(str) + "\nSalt: " + df["Salting"].astype(str) + "\n" + df["Strategy"].astype(str)
    )

    fig, ax1 = plt.subplots(figsize=(13, 6))

    x = np.arange(len(df))
    width = 0.35

    color_exec = "#2b5c8f"  # Steel Blue
    color_skew = "#d95f02"  # Vermillion / Burnt Orange

    # Primary Y-Axis: Execution Time (Bars)
    rects1 = ax1.bar(
        x - width / 2,
        df["Exec_Time_Sec"],
        width,
        label="Execution Time (s)",
        color=color_exec,
        alpha=0.85,
    )
    ax1.set_xlabel("Factorial Run Configurations", labelpad=10, weight="bold")
    ax1.set_ylabel("Execution Time (Seconds)", color=color_exec, weight="bold")
    ax1.tick_params(axis="y", labelcolor=color_exec)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Config_Label"], rotation=0, ha="center")
    ax1.set_ylim(0, max(df["Exec_Time_Sec"].max() * 1.25, 1.0))

    # Secondary Y-Axis: Skew Ratio (Bars)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x + width / 2,
        df["Skew_Ratio_Num"],
        width,
        label="Skew Ratio (x)",
        color=color_skew,
        alpha=0.85,
    )
    ax2.set_ylabel("Skew Ratio (Max / Avg Task Time)", color=color_skew, weight="bold")
    ax2.tick_params(axis="y", labelcolor=color_skew)
    ax2.grid(False)  # Avoid overlapping gridlines
    ax2.set_ylim(0, max(df["Skew_Ratio_Num"].max() * 1.25, 2.0))
    ax2.axhline(1.0, color="green", linestyle="--", alpha=0.6, label="Ideal Skew (1.0x)")

    # Data Values Above Execution Time Bars
    for rect in rects1:
        height = rect.get_height()
        if pd.notna(height) and height > 0:
            ax1.annotate(
                f"{height:.2f}s",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=color_exec,
                weight="bold",
            )

    # Data Values Above Skew Ratio Bars
    for rect in rects2:
        height = rect.get_height()
        if pd.notna(height) and height > 0:
            ax2.annotate(
                f"{height:.2f}x",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=color_skew,
                weight="bold",
            )

    plt.title(
        f"2^3 Factorial Runs: Execution Time vs Partition Skew Ratio [{dataset_type.upper()} DATASET]",
        pad=15,
        weight="bold",
    )

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"  [Plot Saved] -> {file_path}")
    return file_path


def plot_factor_averages(
    results: List[Dict[str, Any]],
    dataset_type: str,
    output_dir: Union[str, Path] = "reports/figures",
) -> Path:
    """Plot 2: Factor Averages (AQE, Salting, Strategy) across Execution Time, Shuffle Read & Skew Ratio."""
    setup_plot_style()
    df = _clean_numeric_data(results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"factor_level_averages_{dataset_type.lower()}.png"
    file_path = output_path / filename

    # Compute Factor Averages
    factors = ["AQE", "Salting", "Strategy"]
    metrics = [
        ("Exec_Time_Sec", "Avg Exec Time (s)"),
        ("Shuffle_Read_MB", "Avg Shuffle Read (MB)"),
        ("Skew_Ratio_Num", "Avg Skew Ratio (x)"),
    ]

    agg_data = []
    for factor in factors:
        if factor in df.columns:
            grouped = df.groupby(factor)[["Exec_Time_Sec", "Shuffle_Read_MB", "Skew_Ratio_Num"]].mean().reset_index()
            for _, row in grouped.iterrows():
                agg_data.append({
                    "Factor": factor,
                    "Level": str(row[factor]),
                    "Exec_Time_Sec": float(row["Exec_Time_Sec"]),
                    "Shuffle_Read_MB": float(row["Shuffle_Read_MB"]),
                    "Skew_Ratio_Num": float(row["Skew_Ratio_Num"]),
                })

    agg_df = pd.DataFrame(agg_data)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)

    for idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[idx]
        
        # Plot bar chart grouped by Factor
        sns.barplot(
            data=agg_df,
            x="Factor",
            y=metric_col,
            hue="Level",
            ax=ax,
            palette="Set2",
        )

        # Annotate exact numerical value on each bar container
        for container in ax.containers:
            for p in container:
                height = p.get_height()
                if pd.notna(height) and height >= 0:
                    fmt = f"{height:.2f}s" if metric_col == "Exec_Time_Sec" else (f"{height:.1f}MB" if metric_col == "Shuffle_Read_MB" else f"{height:.2f}x")
                    ax.annotate(
                        fmt,
                        (p.get_x() + p.get_width() / 2.0, height),
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        xytext=(0, 3),
                        textcoords="offset points",
                        weight="bold",
                    )

        ax.set_title(f"{metric_label}", weight="bold", pad=10)
        ax.set_xlabel("Factor Condition", weight="bold")
        ax.set_ylabel(metric_label, weight="bold")
        ax.set_ylim(0, max(agg_df[metric_col].max() * 1.25, 1.0))
        ax.legend(title="Level", fontsize=8, title_fontsize=9)

    plt.suptitle(
        f"Main Factor Level Averages Across All Runs [{dataset_type.upper()} DATASET]",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot Saved] -> {file_path}")
    return file_path


def generate_all_plots(
    uniform_results: List[Dict[str, Any]],
    skewed_results: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "reports/figures",
) -> List[Path]:
    """Master trigger function to generate all required visual analytics assets."""
    print("\n==========================================================================")
    print("             GENERATING AUTOMATED VISUALIZATION ASSETS                    ")
    print("==========================================================================")

    generated_files = []

    # 1. Dual-Axis Factorial Plots (Uniform & Skewed)
    generated_files.append(
        plot_dual_axis_factorial(uniform_results, dataset_type="Uniform", output_dir=output_dir)
    )
    generated_files.append(
        plot_dual_axis_factorial(skewed_results, dataset_type="Skewed", output_dir=output_dir)
    )

    # 2. Factor Level Average Plots (Uniform & Skewed)
    generated_files.append(
        plot_factor_averages(uniform_results, dataset_type="Uniform", output_dir=output_dir)
    )
    generated_files.append(
        plot_factor_averages(skewed_results, dataset_type="Skewed", output_dir=output_dir)
    )

    print(f"\nSuccessfully generated {len(generated_files)} figures in '{output_dir}/'.\n")
    return generated_files