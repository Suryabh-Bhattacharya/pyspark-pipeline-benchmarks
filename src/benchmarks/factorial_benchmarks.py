"""Factorial Design Benchmark Module with Deep Metrics Capture.

Executes a 2^3 full factorial design experiment across:
  - Factor A: AQE Status (OFF vs ON)
  - Factor B: Salting Status (OFF vs ON - 16 Bins)
  - Factor C: Base Join Strategy (Sort-Merge Join vs Broadcast Hash Join)

Evaluated separately across Uniform and Skewed datasets.
"""

import argparse
from pathlib import Path
import os
import sys
import time
from itertools import product
from typing import Dict, Any, List, Tuple
from tabulate import tabulate

# Apply WinUtils path override only when executing on Windows
if os.name == "nt":
    os.environ["HADOOP_HOME"] = "C:/Hadoop"
    os.environ["PATH"] += os.pathsep + "C:/Hadoop/bin"

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

from src.utils.spark_session import get_spark_session
from src.utils.metrics import MetricsCollector
from src.visualization.plots import generate_all_plots


def execute_factorial_run(
    spark: SparkSession,
    collector: MetricsCollector,
    df_tx: DataFrame,
    df_cust: DataFrame,
    use_salting: bool,
    use_bhj: bool,
    salt_bins: int = 16,
) -> Dict[str, Any]:
    """Executes a single treatment combination in the factorial experiment."""
    stages_before = collector.get_completed_stage_ids()

    start_time = time.perf_counter()

    # 1. Apply Salting Transformation if Enabled
    if use_salting:
        df_tx_prep = df_tx.withColumn(
            "salted_customer_id",
            F.concat(
                F.col("customer_id"),
                F.lit("_"),
                (F.rand(seed=42) * salt_bins).cast("int"),
            ),
        )

        salt_array = F.array([F.lit(i) for i in range(salt_bins)])
        df_cust_prep = (
            df_cust.withColumn("salt", F.explode(salt_array))
            .withColumn(
                "salted_customer_id",
                F.concat(F.col("customer_id"), F.lit("_"), F.col("salt")),
            )
            .drop("salt")
        )
        join_key = "salted_customer_id"
    else:
        df_tx_prep = df_tx
        df_cust_prep = df_cust
        join_key = "customer_id"

    # 2. Apply Join Strategy (BHJ vs SMJ)
    if use_bhj:
        df_joined = df_tx_prep.join(
            F.broadcast(df_cust_prep), on=join_key, how="inner"
        )
    else:
        df_joined = df_tx_prep.join(df_cust_prep, on=join_key, how="inner")

    if use_salting:
        df_joined = df_joined.drop("salted_customer_id")

    record_count = df_joined.count()
    end_time = time.perf_counter()

    duration = max(end_time - start_time, 0.001)
    rps = int(record_count / duration)

    metrics = collector.get_job_stage_metrics(stages_before)

    return {
        "duration": duration,
        "rps": rps,
        "shuffle_mb": metrics["shuffle_read_mb"],
        "spill_mb": metrics["memory_spill_mb"],
        "skew_ratio": metrics["skew_ratio"],
        "count": record_count,
    }


def run_dataset_factorial_suite(
    spark: SparkSession,
    collector: MetricsCollector,
    df_tx: DataFrame,
    df_cust: DataFrame,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """Runs the 2^3 factorial matrix for a specific dataset distribution."""
    print(f"\n==========================================================================")
    print(f"            STARTING 2^3 FACTORIAL SUITE: [{dataset_name.upper()} DATA]")
    print(f"==========================================================================\n")

    aqe_factors = [("OFF", "false"), ("ON", "true")]
    salting_factors = [("OFF", False), ("ON", True)]
    strategy_factors = [("SMJ", False), ("BHJ", True)]

    results: List[Dict[str, Any]] = []
    run_counter = 1

    for (aqe_label, aqe_setting), (salting_label, use_salting), (strategy_label, use_bhj) in product(
        aqe_factors, salting_factors, strategy_factors
    ):
        print(
            f"Run {run_counter}/8: [AQE: {aqe_label}] | [Salting: {salting_label}] | [Strategy: {strategy_label}]..."
        )

        # Clear cached state before each run
        spark.catalog.clearCache()

        # Configure Spark Session for AQE factor
        spark.conf.set("spark.sql.adaptive.enabled", aqe_setting)
        if aqe_setting == "true":
            spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        else:
            spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")

        metrics = execute_factorial_run(
            spark=spark,
            collector=collector,
            df_tx=df_tx,
            df_cust=df_cust,
            use_salting=use_salting,
            use_bhj=use_bhj,
            salt_bins=16,
        )

        print(
            f"  -> Completed in {metrics['duration']:.3f}s | RPS: {metrics['rps']:,} | Skew: {metrics['skew_ratio']}x\n"
        )

        results.append(
            {
                "Run": f"#{run_counter}",
                "Dataset": dataset_name,
                "AQE": aqe_label,
                "Salting": salting_label,
                "Strategy": strategy_label,
                "Exec Time (s)": f"{metrics['duration']:.3f}s",
                "Throughput (RPS)": f"{metrics['rps']:,}",
                "Shuffle Read": f"{metrics['shuffle_mb']} MB",
                "Memory Spill": f"{metrics['spill_mb']} MB",
                "Skew Ratio": f"{metrics['skew_ratio']}x",
            }
        )
        run_counter += 1

    return results


def run_factorial_benchmarks(data_dir: str = "data/raw") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
            print(f"Error: Required dataset '{name}' not found at path: {path.resolve()}")
            sys.exit(1)

    spark = get_spark_session(app_name="FactorialBenchmark")
    collector = MetricsCollector()

    # Disable auto-broadcast threshold so explicit join hints control strategy behavior
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    df_cust = spark.read.parquet(str(cust_path))
    df_tx_uniform = spark.read.parquet(str(tx_uniform_path))
    df_tx_skewed = spark.read.parquet(str(tx_skewed_path))

    # Run Suite 1: Uniform Dataset
    uniform_results = run_dataset_factorial_suite(
        spark, collector, df_tx_uniform, df_cust, "Uniform"
    )

    # Run Suite 2: Skewed Dataset
    skewed_results = run_dataset_factorial_suite(
        spark, collector, df_tx_skewed, df_cust, "Skewed"
    )

    # Print Summary Results Tables
    headers = list(uniform_results[0].keys())

    print("\n==========================================================================")
    print("                UNIFORM DATASET - FACTORIAL RESULTS                      ")
    print("==========================================================================")
    print(tabulate([list(r.values()) for r in uniform_results], headers=headers, tablefmt="grid"))

    print("\n==========================================================================")
    print("                SKEWED DATASET - FACTORIAL RESULTS                       ")
    print("==========================================================================")
    print(tabulate([list(r.values()) for r in skewed_results], headers=headers, tablefmt="grid"))

    return uniform_results, skewed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PySpark Factorial Benchmarks.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing parquet datasets.",
    )
    args = parser.parse_args()

    # 1. Run Factorial Benchmarks and capture in-memory metrics
    uniform_results, skewed_results = run_factorial_benchmarks(data_dir=args.data_dir)

    # 2. Trigger automated plot generation
    generate_all_plots(
        uniform_results=uniform_results,
        skewed_results=skewed_results,
        output_dir="reports/figures",
    )