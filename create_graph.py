import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    'benchmark_result.csv'
)

# ======================
# EXECUTION TIME
# ======================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    df['dataset'],
    df['serial_time'],
    marker='o',
    label='Serial'
)

plt.plot(
    df['dataset'],
    df['parallel_time'],
    marker='o',
    label='Parallel Batch'
)

plt.plot(
    df['dataset'],
    df['thread_time'],
    marker='o',
    label='ThreadPool'
)

plt.xlabel(
    'Dataset Size'
)

plt.ylabel(
    'Seconds'
)

plt.title(
    'Execution Time Comparison'
)

plt.legend()

plt.grid()

plt.savefig(
    'execution_time.png'
)

plt.close()

# ======================
# SPEEDUP
# ======================

x=np.arange(len(df))

width=0.35

plt.figure(
    figsize=(9,5)
)

plt.bar(
    x-width/2,
    df['parallel_speedup'],
    width,
    label='Parallel Batch'
)

plt.bar(
    x+width/2,
    df['thread_speedup'],
    width,
    label='ThreadPool'
)

plt.xticks(
    x,
    df['dataset']
)

plt.xlabel(
    'Dataset Size'
)

plt.ylabel(
    'Speedup'
)

plt.title(
    'Speedup Analysis'
)

plt.legend()

plt.grid(
    axis='y'
)

plt.savefig(
    'speedup.png'
)

plt.close()

# ======================
# THROUGHPUT
# ======================

x=np.arange(len(df))

width=0.25

plt.figure(
    figsize=(10,5)
)

plt.bar(
    x-width,
    df['serial_throughput(img/s)'],
    width,
    label='Serial'
)

plt.bar(
    x,
    df['parallel_throughput(img/s)'],
    width,
    label='Parallel Batch'
)

plt.bar(
    x+width,
    df['thread_throughput(img/s)'],
    width,
    label='ThreadPool'
)

plt.xticks(
    x,
    df['dataset']
)

plt.xlabel(
    'Dataset Size'
)

plt.ylabel(
    'Images / Second'
)

plt.title(
    'Throughput Comparison'
)

plt.legend()

plt.grid(
    axis='y'
)

plt.savefig(
    'throughput.png'
)

plt.close()

print(
    'Visualization completed.'
)