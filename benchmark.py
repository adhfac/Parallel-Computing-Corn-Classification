import os
import time
import pandas as pd

from utils.serial_predict import serial_prediction
from utils.parallel_predict import parallel_prediction
from utils.thread_predict import thread_prediction

datasets = [
    '1000',
    '2000',
    '3000'
]

results = []

for ds in datasets:
    folder = f"data_test/{ds}"
    image_paths = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.lower().endswith(
            (
                '.jpg',
                '.png',
                '.jpeg'
            )
        )
    ]
    total_images = len(
        image_paths
    )
    print(
        f"\nRunning dataset {ds}..."
    )
    # ======================
    # SERIAL
    # ======================

    start = time.time()
    serial_results = serial_prediction(
        image_paths
    )
    serial_time = (
        time.time() - start
    )
    # ======================
    # BATCH PARALLEL
    # ======================
    start = time.time()
    parallel_results = parallel_prediction(
        image_paths
    )
    parallel_time = (
        time.time() - start
    )

    # ======================
    # THREADPOOL
    # ======================
    start = time.time()
    thread_results = thread_prediction(
        image_paths
    )
    thread_time = (
        time.time() - start
    )

    # ======================
    # SPEEDUP
    # ======================
    parallel_speedup = (
        serial_time /
        parallel_time
    )

    thread_speedup = (
        serial_time /
        thread_time
    )

    # ======================
    # THROUGHPUT
    # ======================
    serial_throughput = (
        total_images /
        serial_time
    )

    parallel_throughput = (

        total_images /
        parallel_time
    )

    thread_throughput = (

        total_images /
        thread_time
    )

    # ======================
    # CONSISTENCY CHECK
    # ======================

    parallel_consistency = (

        serial_results ==
        parallel_results
    )

    thread_consistency = (

        serial_results ==
        thread_results
    )

    # ======================
    # SAVE RESULT
    # ======================

    results.append({

        'dataset': ds,

        'num_images':
            total_images,

        'serial_time':
            serial_time,

        'parallel_time':
            parallel_time,

        'thread_time':
            thread_time,

        'parallel_speedup':
            parallel_speedup,

        'thread_speedup':
            thread_speedup,

        'serial_throughput(img/s)':

            serial_throughput,

        'parallel_throughput(img/s)':

            parallel_throughput,

        'thread_throughput(img/s)':

            thread_throughput,

        'parallel_consistency':

            parallel_consistency,

        'thread_consistency':

            thread_consistency
    })

df = pd.DataFrame(
    results
)

print(
    "\n===== BENCHMARK RESULT ====="
)

print(df)

df.to_csv(
    'benchmark_result.csv',
    index=False
)

print(
    "\nSaved to benchmark_result.csv"
)