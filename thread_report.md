# Gambaran Besar

Kode ini melakukan:

> **Klasifikasi banyak gambar menggunakan multithreading (ThreadPool).**

Ide utamanya:

Daripada memproses gambar:

```txt
gambar1 → selesai
gambar2 → selesai
gambar3 → selesai
```

kita memakai beberapa thread supaya:

```txt
Thread1 → gambar1
Thread2 → gambar2
Thread3 → gambar3
Thread4 → gambar4
```

berjalan **bersamaan**.

---

# 1. Import Library

```python
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
from multiprocessing.pool import ThreadPool
import numpy as np
```

Fungsinya:

| Library          | Fungsi                 |
| ---------------- | ---------------------- |
| load_model       | memuat CNN             |
| preprocess_image | preprocessing gambar   |
| ThreadPool       | membuat multithreading |
| numpy            | operasi array          |

---

# 2. Load Model Sekali

```python
MODEL = load_model(
    'model/corn_disease_model.h5',
    compile=False
)
```

Di sini model CNN dimuat **hanya satu kali**.

Misalnya:

```txt
MobileNetV2
```

masuk ke memori.

---

Kenapa ini penting?

Karena pada ThreadPool:

```txt
semua thread berbagi memory
```

jadi **tidak perlu load ulang model**.

---

Visualisasi:

Tanpa shared model:

```txt
Thread1 → load model
Thread2 → load model
Thread3 → load model
```

boros.

---

Dengan kode ini:

```txt
Load model sekali
        ↓
Shared oleh seluruh thread
```

lebih efisien.

---

# 3. Daftar Label

```python
classes = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy'
]
```

Digunakan untuk mengubah output CNN.

Contoh:

```txt
0 → Blight
1 → Common_Rust
2 → Gray_Leaf_Spot
3 → Healthy
```

---

# 4. Fungsi `predict_single()`

Ini fungsi yang memproses **1 gambar**.

---

## Step 1 — Preprocessing

```python
img = preprocess_image(path)
```

Gambar dipersiapkan.

Biasanya:

```txt
resize 224×224
↓
normalize
↓
tensor conversion
```

---

## Step 2 — Prediksi CNN

```python
prediction = MODEL.predict(
    img,
    verbose=0
)
```

Model melakukan inferensi.

Misalnya output:

```txt
[
0.02,
0.85,
0.10,
0.03
]
```

Ini probabilitas kelas.

---

## Step 3 — Ambil nilai terbesar

```python
class_index=np.argmax(
    prediction
)
```

Output:

```txt
1
```

artinya probabilitas terbesar ada di index 1.

---

## Step 4 — Konversi menjadi label

```python
classes[class_index]
```

Hasil:

```txt
Common_Rust
```

---

## Step 5 — Return hasil

```python
return {

    'image':path,

    'label':'Common_Rust'

}
```

Output final:

```python
{
    "image":"leaf1.jpg",
    "label":"Common_Rust"
}
```

---

Flow `predict_single()`:

```txt
Input Gambar
      ↓
Preprocessing
      ↓
CNN Predict
      ↓
Argmax
      ↓
Label
      ↓
Output
```

---

# 5. Fungsi `thread_prediction()`

Ini inti paralelismenya.

---

## Input

```python
thread_prediction(
    image_paths,
    workers=4
)
```

Misalnya:

```txt
100 gambar
4 workers
```

---

## Membuat ThreadPool

```python
with ThreadPool(
    processes=workers
)
```

Artinya:

```txt
buat 4 thread
```

Visualisasi:

```txt
Main Process
     │
 ┌───┼─────────────┐
 │   │     │       │
T1  T2    T3      T4
```

---

## Menjalankan Paralel

```python
results = pool.map(
    predict_single,
    image_paths
)
```

Ini bagian terpenting.

---

Tanpa thread:

```txt
predict(gambar1)

predict(gambar2)

predict(gambar3)
```

serial.

---

Dengan thread:

```txt
Thread1 → predict(gambar1)

Thread2 → predict(gambar2)

Thread3 → predict(gambar3)

Thread4 → predict(gambar4)
```

berjalan **bersamaan**.

---

Pool akan otomatis membagi pekerjaan.

Contoh:

100 gambar.

4 worker.

Distribusi:

```txt
Worker1 → gambar 1–25

Worker2 → gambar 26–50

Worker3 → gambar 51–75

Worker4 → gambar 76–100
```

---

## Menggabungkan hasil

Semua hasil thread dikumpulkan.

```python
return results
```

Output:

```python
[
 {'image':'1.jpg','label':'Blight'},
 {'image':'2.jpg','label':'Healthy'},
 {'image':'3.jpg','label':'Rust'}
]
```

---

# Flowchart sederhana untuk slide

```txt
Input Banyak Gambar
        ↓
Buat ThreadPool (4 Thread)
        ↓
Pool.map()
        ↓
Setiap Thread Jalankan:
    preprocess
    predict
    argmax
        ↓
Gabungkan Hasil
        ↓
Output Prediksi
```

---

# Perbedaan dengan Batch Inference

Ini penting untuk presentasi.

---

### Batch Inference

```txt
Semua gambar
↓
1 predict(batch)
↓
TensorFlow internal parallelism
```

---

### ThreadPool

```txt
Beberapa thread
↓
Masing-masing thread menjalankan
predict_single()
```

---

# Perbedaan dengan Multiprocessing

ThreadPool:

```txt
1 Process
 ├── Thread1
 ├── Thread2
 └── Thread3
```

Shared memory.

**1 model dipakai bersama.**

---

Multiprocessing:

```txt
Process1 → model copy

Process2 → model copy

Process3 → model copy
```

Memory terpisah.

Biasanya RAM lebih besar.

---

# Kalimat sederhana untuk kelas

> _Metode ThreadPool melakukan klasifikasi citra secara paralel menggunakan beberapa thread dalam satu process. Model CNN dimuat satu kali dan dibagikan ke seluruh thread. Setiap thread menjalankan preprocessing serta inferensi terhadap gambar yang berbeda secara bersamaan untuk meningkatkan efisiensi waktu komputasi._
