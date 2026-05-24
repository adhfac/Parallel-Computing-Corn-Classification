Berikut adalah revisi materi presentasi dan penjelasan materi tersebut dengan gaya bahasa yang lebih ilmiah, akademis, dan formal, serta telah disesuaikan agar siap digunakan untuk kebutuhan presentasi di hadapan dosen atau forum akademik.

---

## Analisis Paralelisme pada Proses Inferensi Convolutional Neural Network (CNN)

### 1. Konseptual: Inferensi Serial versus Inferensi Batch (Paralel)

Dalam konteks komputasi dan kecerdasan buatan, pemrosesan data citra dapat dibagi menjadi dua pendekatan utama:

* **Inferensi Serial (Satu per Satu):** Sistem mengeksekusi proses prediksi secara berurutan. Model harus menyelesaikan kalkulasi untuk satu citra sebelum berpindah ke citra berikutnya. Pendekatan ini tidak efisien untuk dataset berskala besar karena menghasilkan beban overhead yang tinggi pada setiap iterasi.
* **Inferensi Batch (Paralelisme Tensor):** Beberapa data citra digabungkan ke dalam satu struktur matriks multidimensi (tensor) dan diumpankan ke model secara simultan. Model memanfaatkan arsitektur perangkat keras (seperti GPU atau CPU modern) untuk memproses seluruh data tersebut dalam satu operasi *forward pass*.

---

### 2. Analisis Struktur Kode Eksisting

#### A. Pemuatan Pustaka (Library) dan Dependensi

```python
from multiprocessing import Pool
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
import numpy as np

```

| Modul / Fungsi | Peranan Akademis |
| --- | --- |
| `load_model` | Memuat arsitektur dan bobot (*weights*) model CNN yang telah dilatih ke dalam memori kerja. |
| `preprocess_image` | Melakukan transformasi visual (seperti penyelarasan dimensi dan normalisasi nilai piksel). |
| `numpy` | Mengelola manipulasi matriks dan operasi aljabar linier pada dataset. |
| `Pool` | *Catatan Teknis:* Fungsi ini diimpor namun **tidak dieksekusi** dalam alur program. |

#### B. Representasi Definitif Kelas Target

```python
classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

```

Array ini berfungsi sebagai fungsi pemetaan (*mapping*) vektor kontraktual yang mengubah indeks probabilitas tertinggi (output numerik dari lapisan *Softmax*) menjadi label taksonomi penyakit tanaman yang bersesuaian.

---

### 3. Evaluasi Metodologi Fungsi `predict_single()` (Pendekatan Serial)

Fungsi `predict_single` mengimplementasikan paradigma eksekusi linier:

1. **Instansiasi Model:** `load_model()` dipanggil pada setiap eksekusi, yang memicu pemborosan latensi karena model dimuat ulang berulang kali ke memori.
2. **Pra-pemrosesan Citra:** Mengubah resolusi spasial menjadi $224 \times 224$ piksel, melakukan normalisasi skala ke rentang $[0, 1]$, dan mengonversinya menjadi format tensor tunggal.
3. **Inferensi Matriks:** `model.predict(img)` mengeksekusi operasi kalkulasi bobot untuk satu sampel.
4. **Reduksi Dimensi:** Fungsi `np.argmax()` mengidentifikasi indeks dengan nilai probabilitas tertinggi pada vektor keluaran untuk menentukan estimasi kelas penyakit.

```
[Citra Masukan] ──> [Pra-pemrosesan] ──> [Inferensi CNN] ──> [Vektor Output] ──> [Klasifikasi Terpilih]

```

---

### 4. Evaluasi Metodologi Fungsi `parallel_prediction()` (Pendekatan Batch)

Fungsi ini menerapkan prinsip *Data Parallelism* pada tingkat framework TensorFlow melalui mekanisme akumulasi *array*.

```python
predictions = model.predict(batch)

```

* **Efisiensi Alokasi Memori:** Fungsi `load_model()` hanya dieksekusi satu kali di awal proses, meminimalkan redundansi I/O memori.
* **Konstruksi Tensor Batch:** Melalui iterasi, citra individual dikumpulkan ke dalam list dan dikonversi menggunakan `np.array(images)` menjadi tensor berdimensi empat dengan format shape:

$$\text{Shape} = (N, H, W, C)$$


> *Di mana $N$ mewakili jumlah sampel (batch size), $H$ dan $W$ adalah dimensi spasial ($224 \times 224$), dan $C$ adalah kanal warna ($3$ untuk RGB).*


* **Optimasi Komputasi Forward Pass:** Berbeda dengan metode serial yang membutuhkan $N$ kali pemanggilan *forward pass*, metode ini mereduksi proses menjadi **1 kali forward pass** saja.

```
Metode Serial:
Predict(img1) ──> Predict(img2) ──> Predict(img3) ──> Predict(img4)  [4 Operasi Terpisah]

Metode Batch:
Predict([img1, img2, img3, img4])                                    [1 Operasi Terpadu]

```

---

### 5. Alur Logika Pemrosesan Data (Flowchart)

```
[Dataset Citra Masukan]
          │
          ▼
   [Pra-pemrosesan] ───> (Penyelarasan Dimensi & Normalisasi)
          │
          ▼
[Akumulasi Tensor Batch] ───> (Matriks Dimensi: N × 224 × 224 × 3)
          │
          ▼
[Inferensi Kolektif] ───> (Eksekusi Paralel via TensorFlow)
          │
          ▼
 [Reduksi Argmax] ───> (Ekstraksi Indeks Nilai Maksimum)
          │
          ▼
[Output Vektor Label] ───> (Hasil Prediksi Terklasifikasi)

```

---

### 6. Narasi Penjelasan Ilmiah untuk Sidang/Presentasi Akademik

> "Berdasarkan analisis struktur kode, mekanisme paralelisasi yang diimplementasikan pada sistem ini tidak berbasis pada pemrograman multi-inti tingkat CPU (*CPU Multiprocessing*), melainkan memanfaatkan teknik **Batch Parallel Inference** yang dioptimalkan secara internal oleh framework TensorFlow.
> Pada metode serial, latensi meningkat secara linier ($O(N)$) seiring bertambahnya volume data karena model melakukan proses *load* dan *forward pass* untuk setiap citra secara individual. Sebaliknya, metode *batch inference* mengonsolidasikan seluruh sampel ke dalam struktur tensor multidimensional sebelum dieksekusi dalam satu siklus *forward pass*.
> Pendekatan ini secara signifikan mereduksi beban *overhead* komputasi dan memaksimalkan utilisasi arsitektur perangkat keras (seperti instruksi SIMD pada CPU atau paralelisasi masif pada GPU), sehingga menghasilkan operasi inferensi yang jauh lebih efisien."

---

### 7. Kesimpulan Teknis Signifikan

* **Klarifikasi Arsitektur:** Keberadaan fungsi `from multiprocessing import Pool` pada deklarasi awal tidak merepresentasikan model komputasi yang berjalan. Tidak ada proses *forking* atau pembuatan *subprocess* berbasis CPU dalam eksekusi kode ini.
* **Terminologi Tepat:** Istilah ilmiah dan akademis yang paling representatif untuk menjelaskan fenomena percepatan komputasi pada kode ini adalah **Vectorized Batch Inference** atau **Tensor Parallelism via TensorFlow**, bukan *CPU Multiprocessing*.