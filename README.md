# PySpark Pipeline Benchmarks

A modular, reproducible PySpark performance benchmarking suite designed to measure the execution impact of common optimization techniques across varying dataset sizes and skew distributions.

This repository implements a $2^3$ factorial experiment matrix analyzing the performance trade-offs of salting, broadcast joins, and dynamic caching in PySpark standalone and multi-node Docker environments.

---

## Repository Structure

```text
pyspark-pipeline-benchmarks/
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── README.md
├── data/
│   └── raw/               # Output directory for synthetic parquet datasets
├── reports/
│   └── figures/           # Generated performance charts and plots 
├── src/
│   ├── data/
│   │   └── generate_datasets.py   # Synthetic data generator for uniform & skewed distributions
│   ├── utils/
│   │   ├── spark_session.py       # Spark session builder with local/cluster auto-detection
│   │   └── metrics.py             # Unified Spark REST API metrics collector
│   ├── visualization/
│   │   └── plot_results.py        # Charting and figure export utilities
│   └── benchmarks/
│       ├── salting_benchmark.py     # Salted join execution pipeline for handling skew
│       ├── join_benchmark.py        # Broadcast join execution pipeline
│       ├── aqe_benchmark.py         # DataFrame caching and persist pipeline
│       └── factorial_benchmark.py   # Full 2^3 factorial matrix runner combining all approaches
└── tests/                         # Unit and integration test suite

## Core Features

* **Isolated Approach Benchmarks:** Dedicated execution pipelines to isolate and evaluate single optimization strategies (Baseline, Key Salting, Broadcast Join and Caching).
* **Factorial Matrix Runner:** Systematically evaluates combination impacts across all $2^3$ treatment conditions to detect interaction effects.
* **Synthetic Data Generator:** Configurable dataset generation with tuneable data volume and skew parameters.
* **Dual Execution Modes:** Runs seamlessly on local standalone PySpark instances or multi-node Docker Spark clusters.
* **Automated Visualizations:** Automatically records benchmark execution times, memory usage, and partition distributions into exportable performance charts.
