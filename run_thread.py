import os
import time
import logging
import threading
import psutil

from utils.thread_predict import thread_prediction

# ======================
# LOGGER CONFIG
# ======================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

datasets = ['1000','2000','3000']

# ======================
# CPU MONITOR
# ======================

cpu_readings = []
monitoring = False

def monitor_cpu():

    global monitoring

    while monitoring:

        cpu = psutil.cpu_percent(interval=0.5)
        cpu_readings.append(cpu)

# ======================
# BENCHMARK START
# ======================

logger.info("="*70)
logger.info("THREADPOOL BENCHMARK STARTED")
logger.info("="*70)

TOTAL_DATASET = len(datasets)

for idx, ds in enumerate(datasets,start=1):

    logger.info("")
    logger.info("="*70)

    logger.info(
        f"RUNNING DATASET [{idx}/{TOTAL_DATASET}] → DATASET {ds}"
    )

    logger.info("="*70)

    dataset_total_start = time.time()

    # ======================
    # PHASE 1
    # ======================

    logger.info(
        f"[DATASET {ds}] Phase 1/3 → Loading dataset..."
    )

    folder = f"data_test/{ds}"

    image_paths = [

        os.path.join(folder,file)

        for file in os.listdir(folder)

        if file.lower().endswith(
            ('.jpg','.png','.jpeg')
        )
    ]

    total_images = len(image_paths)

    logger.info(
        f"[DATASET {ds}] Loaded {total_images} images."
    )

    # ======================
    # PHASE 2
    # ======================

    logger.info(
        f"[DATASET {ds}] Phase 2/3 → "
        f"Starting THREADPOOL inference..."
    )

    logger.info(
        f"[DATASET {ds}] Thread workers ACTIVE"
    )

    logger.info(
        f"[DATASET {ds}] Concurrent execution STARTED"
    )

    cpu_readings.clear()

    monitoring = True

    cpu_thread = threading.Thread(
        target=monitor_cpu
    )

    cpu_thread.start()

    start = time.time()

    results = thread_prediction(
        image_paths
    )

    inference_time = (
        time.time() - start
    )

    monitoring = False

    cpu_thread.join()

    logger.info(
        f"[DATASET {ds}] Thread inference FINISHED"
    )

    # ======================
    # PHASE 3
    # ======================

    throughput = (
        total_images /
        inference_time
    )

    avg_cpu = (
        sum(cpu_readings) /
        len(cpu_readings)
        if cpu_readings else 0
    )

    peak_cpu = (
        max(cpu_readings)
        if cpu_readings else 0
    )

    total_elapsed = (
        time.time() -
        dataset_total_start
    )

    logger.info(
        f"[DATASET {ds}] Execution Time : "
        f"{inference_time:.2f} sec"
    )

    logger.info(
        f"[DATASET {ds}] Throughput : "
        f"{throughput:.2f} img/s"
    )

    logger.info(
        f"[DATASET {ds}] Avg CPU Usage : "
        f"{avg_cpu:.2f}%"
    )

    logger.info(
        f"[DATASET {ds}] Peak CPU Usage : "
        f"{peak_cpu:.2f}%"
    )

    logger.info(
        f"[DATASET {ds}] TOTAL DATASET TIME : "
        f"{total_elapsed:.2f} sec"
    )

    logger.info(
        f"[DATASET {ds}] STATUS → COMPLETED ✓"
    )

logger.info("")
logger.info("="*70)
logger.info("THREADPOOL BENCHMARK COMPLETED")
logger.info("="*70)