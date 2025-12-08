import asyncio
import json
import os
import sys
import subprocess
import time
from datetime import datetime

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # d:\AI\xiaoyou-core
MVP_CORE_DIR = os.path.join(ROOT_DIR, "mvp_core")
EXPERIMENTS_DIR = os.path.join(MVP_CORE_DIR, "experiments")
SCRIPT_PATH = os.path.join(EXPERIMENTS_DIR, "comprehensive_experiment.py")
OUTPUT_DIR = os.path.join(MVP_CORE_DIR, "experiment_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_experiment(mode, workload, output_file):
    print(f"Running experiment: mode={mode}, workload={workload}...")
    cmd = [sys.executable, SCRIPT_PATH, "--mode", mode, "--workload", workload, "--output", output_file]
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished {mode}/{workload}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {mode}/{workload}: {e}")

def main():
    # Define the 3 categories requested
    experiments = [
        ("naive_async", "mock", "naive_mock.json"),  # Traditional Async Mock
        ("xy_core", "mock", "xy_core_mock.json"),    # My Architecture Mock
        ("xy_core", "real", "xy_core_real.json")     # My Architecture Real
    ]

    for mode, workload, filename in experiments:
        output_path = os.path.join(OUTPUT_DIR, filename)
        run_experiment(mode, workload, output_path)

    print("\nAll experiments completed.")
    
    # Run Visualization
    print("Generating charts...")
    viz_script = os.path.join(EXPERIMENTS_DIR, "visualize_benchmark.py")
    subprocess.run([sys.executable, viz_script], cwd=EXPERIMENTS_DIR, check=True)
    
    print("Charts generated in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
