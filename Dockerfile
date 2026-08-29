FROM python:3.11-slim

# Install Java 17 OpenJDK and system utilities required by PySpark
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java Environment Variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set Container Working Directory
WORKDIR /app

# Copy Requirements first to utilize Docker build layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Set Python Path so module imports resolve correctly inside container
ENV PYTHONPATH=/app

# Default command: Runs the complete factorial benchmark suite
CMD ["python", "-m", "src.benchmarks.factorial_benchmarks"]
