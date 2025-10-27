# PDC Lab Exam - Parallel Image Processing

This project implements and compares three different methods for batch-processing images using Python, as part of the CSC334 Parallel and Distributed Computing lab exam.

The scripts process a dataset of 57 images across 4 folders, resize them to 128x128, and add a watermark.

## Scripts

* `sequential_process.py`: Processes all images one by one in a single thread.
* `parallel_process.py`: Uses `multiprocessing.Pool` to process images in parallel using 1, 2, 4, and 8 workers and prints a speedup table.
* `distributed_sim.py`: Simulates a distributed environment by splitting the work between 2 worker processes (`Nodes`) and reports their individual and total times.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repo-URL]
    cd [Your-Repo-Name]
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python-headless numpy
    ```

3.  **Ensure dataset is present:**
    Make sure the `images_dataset` folder (containing subfolders `cats`, `cars`, `dogs`, `flowers`) is in the same directory.

4.  **Run the scripts:**
    ```bash
    # Run the sequential script
    python sequential_process.py

    # Run the parallel comparison
    python parallel_process.py

    # Run the distributed simulation
    python distributed_sim.py
    ```

## Results

The processing task on this dataset was extremely fast (0.34s sequentially). Due to this, the overhead of creating and managing processes was greater than the computational savings, leading to a slowdown in parallel versions.

### Parallel Speedup Table

| Workers | Time (s) | Speedup |
| :--- | :--- | :--- |
| 1 | 0.95 | 1.00x |
| 2 | 0.83 | 1.16x |
| 4 | 1.28 | 0.74x |
| 8 | 1.98 | 0.48x |

### Distributed Simulation Summary

* Node 1 processed 28 images in 0.3s
* Node 2 processed 29 images in 0.3s
* **Total distributed time: 1.1s**
* **Efficiency: 0.28x over sequential**

## Analysis

* **Best Configuration:** 2 workers (0.83s) was the fastest *parallel* run. However, the **sequential script (0.34s)** was the fastest overall.
* **Bottleneck:** The primary bottleneck was **parallel overhead**, not computation. The task workload was too small for parallelism to be effective. The time to create, manage, and sync processes was far greater than the time saved.
