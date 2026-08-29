"""SparkSession builder utility with cross-platform compatibility and cluster support."""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "PySpark-Distributed-Benchmark",
    master_url: str | None = None,
) -> SparkSession:
    """Creates or gets a SparkSession.

    Defaults to SPARK_MASTER_URL env var, or local[*] if unspecified. Dynamically
    configures Windows environment variables and RawLocalFileSystem fallbacks
    to prevent native Hadoop binary errors across environments.
    """
    # Auto-detect Windows host environment and set up Hadoop fallbacks dynamically
    if sys.platform.startswith("win"):
        default_hadoop_path = Path("C:/Hadoop")
        if "HADOOP_HOME" not in os.environ and default_hadoop_path.exists():
            os.environ["HADOOP_HOME"] = str(default_hadoop_path)
            hadoop_bin = default_hadoop_path / "bin"
            if hadoop_bin.exists():
                os.environ["PATH"] += os.pathsep + str(hadoop_bin)

    # Dedicated project-relative temp directory
    tmp_dir = Path("./tmp/spark").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if master_url is None:
        master_url = os.getenv("SPARK_MASTER_URL", "local[*]")

    # Resolve Driver Host: Defaults to environment variable if set (Docker), else None (Local)
    driver_host = os.getenv("SPARK_DRIVER_HOST", None)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master_url)
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.local.dir", str(tmp_dir))
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        # Direct raw filesystem bypass to prevent Windows NativeIO$Windows.access0 crashes
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    )

    if driver_host:
        builder = builder.config("spark.driver.host", driver_host)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark