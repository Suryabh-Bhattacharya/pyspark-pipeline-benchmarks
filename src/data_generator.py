"""Synthetic Data Generator module for PySpark Distributed Pipeline Benchmarks.

Generates both uniform baseline and skewed transaction data using
synthetic key allocations to simulate real-world hot-key bottlenecks.
"""

from pathlib import Path
import time
import os

# Set Hadoop Home for Windows local execution
os.environ["HADOOP_HOME"] = "C:/Hadoop"
os.environ["PATH"] += os.pathsep + "C:/Hadoop/bin"

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

from src.utils.spark_session import get_spark_session


def generate_transactions(
    spark: SparkSession,
    num_rows: int = 1_000_000,
    num_partitions: int = 16,
    skew_ratio: float = 0.0,
    hot_key: int = 0,
    seed: int = 42,
) -> DataFrame:
    """Generates a DataFrame of synthetic transactions.

    Args:
        spark: Active SparkSession instance.
        num_rows: Total rows to generate.
        num_partitions: Number of output partitions.
        skew_ratio: Fraction of rows assigned to the single hot_key (0.0 to 1.0).
        hot_key: The ID value that will absorb the skewed proportion of rows.
        seed: Random seed for reproducible dataset generation.

    Returns:
        Spark DataFrame with transaction records.
    """
    skewed_count = int(num_rows * skew_ratio)
    uniform_count = num_rows - skewed_count

    # 1. Generate skewed partition (if skew_ratio > 0)
    if skewed_count > 0:
        df_skewed = (
            spark.range(0, skewed_count, step=1, numPartitions=max(1, num_partitions // 2))
            .withColumn("customer_id", F.lit(hot_key))
        )
    else:
        df_skewed = None

    # 2. Generate uniform partition
    if uniform_count > 0:
        df_uniform = (
            spark.range(0, uniform_count, step=1, numPartitions=max(1, num_partitions // 2))
            .withColumn("customer_id", (F.rand(seed=seed) * 10000 + 1).cast("int"))
        )
    else:
        df_uniform = None

    # 3. Combine parts depending on skew presence
    if df_skewed is not None and df_uniform is not None:
        df_base = df_skewed.union(df_uniform)
    elif df_skewed is not None:
        df_base = df_skewed
    else:
        df_base = df_uniform

    # 4. Attach transaction attributes
    df_transactions = (
        df_base
        .withColumnRenamed("id", "transaction_id")
        .withColumn("amount", F.round(F.rand(seed=seed + 1) * 500.0 + 5.0, 2))
        .withColumn(
            "timestamp",
            F.to_timestamp(
                F.from_unixtime(
                    F.lit(1700000000) + (F.rand(seed=seed + 2) * 8640000).cast("int")
                )
            ),
        )
    )

    return df_transactions.repartition(num_partitions)


def generate_customer_dim(
    spark: SparkSession,
    num_customers: int = 10001,
    num_partitions: int = 4,
) -> DataFrame:
    """Generates a dimension lookup table of unique customer profiles."""
    return (
        spark.range(0, num_customers, 1, num_partitions)
        .withColumnRenamed("id", "customer_id")
        .withColumn("customer_name", F.concat(F.lit("Customer_"), F.col("customer_id")))
        .withColumn(
            "region",
            F.element_at(
                F.array(
                    F.lit("APAC"),
                    F.lit("EMEA"),
                    F.lit("NA"),
                    F.lit("LATAM"),
                ),
                (F.rand(seed=7) * 4 + 1).cast("int"),
            ),
        )
    )


def save_benchmark_datasets(
    output_dir: str = "data/raw",
    num_rows: int = 2_000_000,
    skew_ratio: float = 0.8,
) -> None:
    """Generates and writes both uniform and skewed Parquet datasets to disk."""
    spark = get_spark_session(app_name="DataGenerator")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # 1. Generate & Write Uniform Dataset (0% skew baseline)
    print(f"Generating {num_rows:,} transactions with UNIFORM distribution...")
    df_tx_uniform = generate_transactions(spark, num_rows=num_rows, skew_ratio=0.0)
    tx_uniform_path = str(out_path / "transactions_uniform.parquet")
    df_tx_uniform.write.mode("overwrite").parquet(tx_uniform_path)

    # 2. Generate & Write Skewed Dataset (e.g., 80% skew)
    print(f"Generating {num_rows:,} transactions with {skew_ratio*100:.0f}% SKEW...")
    df_tx_skewed = generate_transactions(spark, num_rows=num_rows, skew_ratio=skew_ratio)
    tx_skewed_path = str(out_path / "transactions_skewed.parquet")
    df_tx_skewed.write.mode("overwrite").parquet(tx_skewed_path)

    # 3. Generate & Write Customer Dimension
    print("Generating Customer dimension lookup table...")
    df_cust = generate_customer_dim(spark)
    cust_path = str(out_path / "customers.parquet")
    df_cust.write.mode("overwrite").parquet(cust_path)

    elapsed = time.time() - start_time
    print(f"\nAll datasets generated successfully in {elapsed:.2f} seconds!")
    print(f"  Uniform Transactions -> {tx_uniform_path}")
    print(f"  Skewed Transactions  -> {tx_skewed_path}")
    print(f"  Customers Dimension  -> {cust_path}")


if __name__ == "__main__":
    save_benchmark_datasets()