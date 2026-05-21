import os
import time

from utils.serial_predict import serial_prediction
from utils.parallel_predict import parallel_prediction

AVAILABLE_DATASET = ['10','50','100','500']

print("\n=== CORN DISEASE CLASSIFICATION ===")

print("\nJumlah dataset tersedia:")

for item in AVAILABLE_DATASET:
    print(f"- {item} gambar")

dataset_choice=input(
    "\nPilih dataset (10/50/100/500): "
)

if dataset_choice not in AVAILABLE_DATASET:

    print("Pilihan tidak valid.")
    exit()

method=input(
    "Metode (serial/parallel): "
).lower()

folder=f"data_test/{dataset_choice}"

image_paths=[]

for file in os.listdir(folder):

    filepath=os.path.join(folder,file)

    if file.lower().endswith(
        ('.jpg','.jpeg','.png')
    ):
        image_paths.append(filepath)

print(f"\nTotal gambar: {len(image_paths)}")

start=time.time()

if method=='serial':

    predictions=serial_prediction(
        image_paths
    )

elif method=='parallel':

    predictions=parallel_prediction(
        image_paths
    )

else:

    print("Metode tidak valid.")
    exit()

end=time.time()

execution_time=end-start

print("\n===== HASIL =====")

for result in predictions:

    print(
        f"{os.path.basename(result['image'])}"
        f" -> {result['label']}"
    )

print("\n===== PERFORMANCE =====")

print(
    f"Metode : {method}"
)

print(
    f"Jumlah Data : {dataset_choice}"
)

print(
    f"Waktu Komputasi : "
    f"{execution_time:.4f} detik"
)