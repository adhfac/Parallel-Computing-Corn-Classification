import os
import time
import pandas as pd

from utils.serial_predict import serial_prediction
from utils.parallel_predict import parallel_prediction

datasets=['10','50','100','500']

results=[]

for ds in datasets:

    folder=f"data_test/{ds}"

    image_paths=[

        os.path.join(folder,file)

        for file in os.listdir(folder)

        if file.endswith(
            ('.jpg','.png','.jpeg')
        )
    ]

    start=time.time()

    serial_prediction(
        image_paths
    )

    serial_time=time.time()-start

    start=time.time()

    parallel_prediction(
        image_paths
    )

    parallel_time=time.time()-start

    speedup=serial_time/parallel_time

    results.append({

        'dataset':ds,
        'serial_time':serial_time,
        'parallel_time':parallel_time,
        'speedup':speedup
    })

df=pd.DataFrame(results)

print(df)

df.to_csv(
    'benchmark_result.csv',
    index=False
)