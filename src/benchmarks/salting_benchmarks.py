"""Salting Technique Benchmark Module with Stage-Aware Deep Metrics Capture.

Demonstrates key salting to fix data skew bottlenecks when joining two large DataFrames 
where Broadcast Hash Join (BHJ) is disabled or impossible. Tests both Uniform and Skewed data.
"""

from pathlib import Path
import time
import os
import sys
from typing import Dict, Any
from tabulate import tabulate

os.environ["HADOOP_HOME"] = "C:/Hadoop"
os.environ["PATH"] += os.pathsep + "C:/Hadoop/bin"

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

from src.utils.spark_session import get_spark_session
from src.utils.metrics import MetricsCollector


def benchmark_standard_smj(
    spark: SparkSession,
    collector: MetricsCollector,
    df_tx: DataFrame,
    df_cust: DataFrame,
    dataset_name: str,
) -> Dict[str, Any]:
    """Executes a standard Sort-Merge Join (Baseline)."""
    print(f"Running Standard SMJ on [{dataset_name} Data]...")

    stages_before = collector.get_completed_stage_ids()

    start_time = time.perf_counter()
    df_joined = df_tx.join(df_cust, on="customer_id", how="inner")
    record_count = df_joined.count()
    end_time = time.perf_counter()

    duration = max(end_time - start_time, 0.001)
    rps = int(record_count / duration)

    metrics = collector.get_job_stage_metrics(stages_before)

    print(
        f"  Completed in {duration:.3f}s | Result count: {record_count:,}\n"
    )

    return {
        "Dataset": dataset_name,
        "Strategy": "Standard SMJ",
        "Exec Time (s)": f"{duration:.3f}s",
        "Throughput (RPS)": f"{rps:,}",
        "Shuffle Read": f"{metrics['shuffle_read_mb']} MB",
        "Memory Spill": f"{metrics['memory_spill_mb']} MB",
        "Skew Ratio": f"{metrics['skew_ratio']}x",
    }


def benchmark_salted_smj(
    spark: SparkSession,
    collector: MetricsCollector,
    df_tx: DataFrame,
    df_cust: DataFrame,
    dataset_name: str,
    salt_bins: int = 16,
) -> Dict[str, Any]:
    """Executes a Salted Sort-Merge Join to break up hot keys."""
    print(
        f"Running Salted SMJ (Salt Bins = {salt_bins}) on [{dataset_name} Data]..."
    )

    stages_before = collector.get_completed_stage_ids()

    start_time = time.perf_counter()

    # 1. Add random salt to Fact Table (Transactions)
    df_tx_salted = df_tx.withColumn(
        "salted_customer_id",
        F.concat(
            F.col("customer_id"),
            F.lit("_"),
            (F.rand(seed=42) * salt_bins).cast("int"),
        ),
    )

    # 2. Replicate Dimension Table keys across all salt bins
    salt_array = F.array([F.lit(i) for i in range(salt_bins)])
    df_cust_salted = (
        df_cust.withColumn("salt", F.explode(salt_array))
        .withColumn(
            "salted_customer_id",
            F.concat(F.col("customer_id"), F.lit("_"), F.col("salt")),
        )
        .drop("salt")
    )

    # 3. Join on the new salted composite key
    df_joined = df_tx_salted.join(
        df_cust_salted, on="salted_customer_id", how="inner"
    ).drop("salted_customer_id")

    record_count = df_joined.count()
    end_time = time.perf_counter()

    duration = max(end_time - start_time, 0.001)
    rps = int(record_count / duration)

    metrics = collector.get_job_stage_metrics(stages_before)

    print(
        f"  Completed in {duration:.3f}s | Result count: {record_count:,}\n"
    )

    return {
        "Dataset": dataset_name,
        "Strategy": f"Salted SMJ ({salt_bins} Bins)",
        "Exec Time (s)": f"{duration:.3f}s",
        "Throughput (RPS)": f"{rps:,}",
        "Shuffle Read": f"{metrics['shuffle_read_mb']} MB",
        "Memory Spill": f"{metrics['memory_spill_mb']} MB",
        "Skew Ratio": f"{metrics['skew_ratio']}x",
    }


def run_salting_benchmarks(data_dir: str = "data/raw") -> None:
    spark = get_spark_session(app_name="SaltingBenchmark")
    collector = MetricsCollector()

    # Disable AQE and Broadcast Join to force pure Sort-Merge Join behavior
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    base_path = Path(data_dir)
    tx_uniform_path = base_path / "transactions_uniform.parquet"
    tx_skewed_path = base_path / "transactions_skewed.parquet"
    cust_path = base_path / "customers.parquet"

    for path, name in [
        (tx_uniform_path, "Uniform Transactions"),
        (tx_skewed_path, "Skewed Transactions"),
        (cust_path, "Customers Lookup"),
    ]:
        if not path.exists():
            print(
                f"Error: Required dataset '{name}' not found at path: {path.resolve()}"
            )
            sys.exit(1)

    df_cust = spark.read.parquet(str(cust_path))
    df_tx_uniform = spark.read.parquet(str(tx_uniform_path))
    df_tx_skewed = spark.read.parquet(str(tx_skewed_path))

    print(
        "=================================================="
    )
    print(
        "     KEY SALTING SKEW MITIGATION BENCHMARK        "
    )
    print(
        "==================================================\n"
    )

    results = []

    # 1. Uniform Dataset Tests
    results.append(
        benchmark_standard_smj(
            spark, collector, df_tx_uniform, df_cust, "Uniform"
        )
    )
    results.append(
        benchmark_salted_smj(
            spark, collector, df_tx_uniform, df_cust, "Uniform", salt_bins=16
        )
    )

    # 2. Skewed Dataset Tests
    results.append(
        benchmark_standard_smj(
            spark, collector, df_tx_skewed, df_cust, "Skewed"
        )
    )
    results.append(
        benchmark_salted_smj(
            spark, collector, df_tx_skewed, df_cust, "Skewed", salt_bins=16
        )
    )

    # Summary Output
    print(
        "=================================================="
    )
    print(
        "             BENCHMARK SUMMARY RESULTS            "
    )
    print(
        "=================================================="
    )
    headers = list(results[0].keys())
    rows = [list(r.values()) for r in results]

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    run_salting_benchmarks()