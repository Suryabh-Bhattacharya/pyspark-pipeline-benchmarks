"""Join Optimization Benchmark Module with Stage-Aware Deep Metrics Capture."""

import argparse
from pathlib import Path
import os
import sys
import time
from typing import Callable, Dict, Any
from tabulate import tabulate

os.environ["HADOOP_HOME"] = "C:/Hadoop"
os.environ["PATH"] += os.pathsep + "C:/Hadoop/bin"

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

from src.utils.spark_session import get_spark_session
from src.utils.metrics import MetricsCollector


def benchmark_join_run(
    spark: SparkSession,
    collector: MetricsCollector,
    transformation_func: Callable[[], DataFrame],
    dataset_name: str,
    strategy_name: str,
) -> Dict[str, Any]:
    """Executes a join benchmark run, tracking exact time and stage-level delta metrics."""
    print(f"Running [{strategy_name}] on [{dataset_name}]...")

    # Capture stage snapshot BEFORE executing action
    stages_before = collector.get_completed_stage_ids()

    start_time = time.perf_counter()
    df = transformation_func()
    record_count = df.count()  # Trigger Spark Action
    end_time = time.perf_counter()

    duration = max(end_time - start_time, 0.001)
    rps = int(record_count / duration)

    # Fetch metrics ONLY for stages completed during this run
    metrics = collector.get_job_stage_metrics(stages_before)

    print(
        f"  Completed in {duration:.3f}s | Result count: {record_count:,}\n"
    )

    return {
        "Dataset": dataset_name,
        "Strategy": strategy_name,
        "Exec Time (s)": f"{duration:.3f}s",
        "Throughput (RPS)": f"{rps:,}",
        "Shuffle Read": f"{metrics['shuffle_read_mb']} MB",
        "Memory Spill": f"{metrics['memory_spill_mb']} MB",
        "Skew Ratio": f"{metrics['skew_ratio']}x",
    }


def run_join_benchmarks(
    data_dir: str = "data/raw",
    tx_uniform_filename: str = "transactions_uniform.parquet",
    tx_skewed_filename: str = "transactions_skewed.parquet",
    cust_filename: str = "customers.parquet",
    join_key: str = "customer_id",
) -> None:
    base_path = Path(data_dir)
    tx_uniform_path = base_path / tx_uniform_filename
    tx_skewed_path = base_path / tx_skewed_filename
    cust_path = base_path / cust_filename

    # Verify input datasets exist prior to running Spark transformations
    for path, name in [
        (tx_uniform_path, "Uniform Transactions"),
        (tx_skewed_path, "Skewed Transactions"),
        (cust_path, "Customers Lookup"),
    ]:
        if not path.exists():
            print(
                f"Error: Required dataset '{name}' not found at path: {path.resolve()}"
            )
            print(
                "Provide valid dataset paths via parameters or place files in data directory."
            )
            sys.exit(1)

    spark = get_spark_session(app_name="JoinBenchmark")
    collector = MetricsCollector()

    # Disable AQE auto-broadcast and auto-skew join for explicit baseline measurement
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    print(
        "=========================================================================="
    )
    print(
        "         PYSPARK JOIN PERFORMANCE BENCHMARK (WITH METRICS)                "
    )
    print(
        f"         Target Join Key: '{join_key}'                                     "
    )
    print(
        "==========================================================================\n"
    )

    # Pre-read DataFrames to isolate join logic from file system reads
    df_tx_uniform = spark.read.parquet(str(tx_uniform_path))
    df_tx_skewed = spark.read.parquet(str(tx_skewed_path))
    df_cust = spark.read.parquet(str(cust_path))

    results = []

    # 1. Uniform - Sort-Merge Join (SMJ)
    res_u_smj = benchmark_join_run(
        spark=spark,
        collector=collector,
        transformation_func=lambda: df_tx_uniform.join(
            df_cust, on=join_key, how="inner"
        ),
        dataset_name="Uniform",
        strategy_name="Sort-Merge Join (SMJ)",
    )
    results.append(res_u_smj)

    # 2. Uniform - Broadcast Hash Join (BHJ)
    res_u_bhj = benchmark_join_run(
        spark=spark,
        collector=collector,
        transformation_func=lambda: df_tx_uniform.join(
            F.broadcast(df_cust), on=join_key, how="inner"
        ),
        dataset_name="Uniform",
        strategy_name="Broadcast Hash Join (BHJ)",
    )
    results.append(res_u_bhj)

    # 3. Skewed - Sort-Merge Join (SMJ)
    res_s_smj = benchmark_join_run(
        spark=spark,
        collector=collector,
        transformation_func=lambda: df_tx_skewed.join(
            df_cust, on=join_key, how="inner"
        ),
        dataset_name="Skewed",
        strategy_name="Sort-Merge Join (SMJ)",
    )
    results.append(res_s_smj)

    # 4. Skewed - Broadcast Hash Join (BHJ)
    res_s_bhj = benchmark_join_run(
        spark=spark,
        collector=collector,
        transformation_func=lambda: df_tx_skewed.join(
            F.broadcast(df_cust), on=join_key, how="inner"
        ),
        dataset_name="Skewed",
        strategy_name="Broadcast Hash Join (BHJ)",
    )
    results.append(res_s_bhj)

    # Display Rich Metrics Summary Table
    print(
        "=========================================================================="
    )
    print(
        "                         JOIN BENCHMARK RESULTS                           "
    )
    print(
        "=========================================================================="
    )
    headers = list(results[0].keys())
    rows = [list(r.values()) for r in results]

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PySpark Join Benchmarks on custom datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing parquet datasets.",
    )
    parser.add_argument(
        "--tx-uniform",
        type=str,
        default="transactions_uniform.parquet",
        help="Uniform dataset filename.",
    )
    parser.add_argument(
        "--tx-skewed",
        type=str,
        default="transactions_skewed.parquet",
        help="Skewed dataset filename.",
    )
    parser.add_argument(
        "--cust",
        type=str,
        default="customers.parquet",
        help="Lookup dataset filename.",
    )
    parser.add_argument(
        "--join-key",
        type=str,
        default="customer_id",
        help="Join key column present across datasets.",
    )

    args = parser.parse_args()

    run_join_benchmarks(
        data_dir=args.data_dir,
        tx_uniform_filename=args.tx_uniform,
        tx_skewed_filename=args.tx_skewed,
        cust_filename=args.cust,
        join_key=args.join_key,
    )