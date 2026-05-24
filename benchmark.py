import os
import time
import pandas as pd
from utils.serial_predict import serial_prediction
from utils.parallel_predict import parallel_prediction

datasets = ['1000', '2000', '3000']
results = []

for ds in datasets:
    folder = f"data_test/{ds}"
    image_paths = [
        os.path.join(folder, file) 
        for file in os.listdir(folder) 
        if file.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    total_images = len(image_paths)
    print(f"\nRunning dataset {ds}...")
    
    # ======================
    # SERIAL
    # ======================
    start = time.time()
    serial_results = serial_prediction(image_paths)
    serial_time = time.time() - start
    
    # ======================
    # PARALLEL
    # ======================
    start = time.time()
    parallel_results = parallel_prediction(image_paths)
    parallel_time = time.time() - start
    
    # ======================
    # SPEEDUP
    # ======================
    speedup = serial_time / parallel_time
    
    # ======================
    # THROUGHPUT
    # ======================
    serial_throughput = total_images / serial_time
    parallel_throughput = total_images / parallel_time
    
    # ======================
    # ACCURACY CONSISTENCY
    # ======================
    consistency_check = (serial_results == parallel_results)
    
    results.append({
        'dataset': ds,
        'num_images': total_images,
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'serial_throughput(img/s)': serial_throughput,
        'parallel_throughput(img/s)': parallel_throughput,
        'consistency_check': consistency_check
    })

df = pd.DataFrame(results)

print("\n===== BENCHMARK RESULT =====")
print(df)

df.to_csv('benchmark_result.csv', index=False)
print("\nSaved to benchmark_result.csv")