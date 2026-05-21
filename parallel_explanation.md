## Implementasi Parallel Processing Menggunakan Batch Inference

Pada penelitian ini, optimisasi komputasi dilakukan menggunakan metode **batch inference** pada proses klasifikasi penyakit daun jagung berbasis **Convolutional Neural Network (CNN)**.

Berbeda dengan metode serial yang melakukan inferensi terhadap satu gambar pada satu waktu, batch inference memungkinkan beberapa gambar diproses secara bersamaan dalam satu kali eksekusi model.

Implementasi metode ini direalisasikan melalui fungsi `parallel_prediction()`.

### 1. Loading Model CNN

Tahap pertama adalah memuat model CNN yang telah dilatih sebelumnya menggunakan TensorFlow/Keras.

```python
model = load_model(
    'model/corn_disease_model.h5'
)
```

Fungsi `load_model()` bertugas mengambil model hasil pelatihan yang tersimpan pada file `.h5`. Model ini berisi arsitektur jaringan, bobot (*weights*), serta parameter pembelajaran yang akan digunakan untuk proses inferensi.

Setelah model berhasil dimuat, sistem siap melakukan klasifikasi citra.

---

### 2. Preprocessing Seluruh Gambar

Tahap berikutnya adalah melakukan preprocessing terhadap seluruh gambar input.

```python
images=[]

for path in image_paths:

    img = preprocess_image(path)[0]

    images.append(img)
```

Pada bagian ini, sistem melakukan iterasi terhadap seluruh path gambar yang tersedia pada dataset.

Setiap gambar diproses menggunakan fungsi `preprocess_image()`.

Tahapan preprocessing umumnya meliputi:

* membaca file gambar,
* resize gambar menjadi ukuran input CNN,
* normalisasi nilai piksel,
* konversi menjadi tensor numerik.

Hasil preprocessing berupa tensor gambar kemudian disimpan ke dalam list `images`.

Penggunaan indeks `[0]` dilakukan untuk menghilangkan dimensi batch tunggal yang sebelumnya ditambahkan saat preprocessing.

---

### 3. Pembentukan Batch Tensor

Setelah seluruh gambar selesai diproses, sistem menggabungkannya menjadi satu tensor batch.

```python
batch=np.array(images)
```

Fungsi `np.array()` mengubah list gambar menjadi sebuah array multidimensi NumPy.

Sebagai contoh:

Jika terdapat **100 gambar** dengan ukuran input:

```txt
224 × 224 × 3
```

maka bentuk tensor batch menjadi:

```txt
(100,224,224,3)
```

dimana:

* **100** → jumlah gambar dalam dataset
* **224** → tinggi citra
* **224** → lebar citra
* **3** → channel warna RGB

Tensor batch ini menjadi input utama untuk proses inferensi paralel.

---

### 4. Batch Inference pada CNN

Tahap inti dari implementasi paralel terdapat pada proses inferensi berikut:

```python
predictions=model.predict(
    batch,
    verbose=0
)
```

Fungsi `model.predict()` menjalankan inferensi terhadap seluruh tensor batch secara bersamaan.

Inilah yang disebut **batch inference**.

Alih-alih menjalankan:

```txt
1 gambar → predict()
1 gambar → predict()
1 gambar → predict()
```

seperti pada metode serial, batch inference melakukan:

```txt
100 gambar → 1 kali predict()
```

TensorFlow kemudian memanfaatkan mekanisme komputasi paralel internal pada operasi CNN, seperti:

* operasi matriks (*matrix multiplication*)
* convolution operation
* tensor computation
* SIMD CPU optimization
* GPU kernel parallelization (jika tersedia)

Dengan demikian, beberapa gambar dapat diproses dalam satu siklus inferensi.

Keuntungan utama pendekatan ini adalah mengurangi overhead pemanggilan model berulang serta meningkatkan utilisasi sumber daya komputasi.

---

### 5. Pengambilan Hasil Prediksi

Setelah inferensi selesai dilakukan, sistem mengambil kelas prediksi dari setiap output model.

```python
results=[]

for i,pred in enumerate(predictions):

    class_index=np.argmax(pred)
```

Output CNN berupa probabilitas untuk setiap kelas.

Sebagai contoh:

```txt
[0.02, 0.91, 0.05, 0.02]
```

Nilai probabilitas terbesar dipilih menggunakan fungsi `np.argmax()`.

Pada contoh di atas:

```txt
index = 1
```

yang merepresentasikan kelas:

```txt
Common_Rust
```

---

### 6. Penyusunan Output

Tahap terakhir adalah menyusun hasil klasifikasi.

```python
results.append({
    'image': image_paths[i],
    'label': classes[class_index]
})
```

Sistem menyimpan:

* nama/path gambar
* label hasil prediksi

ke dalam list hasil (`results`).

Kemudian seluruh hasil dikembalikan melalui:

```python
return results
```

---

## Fungsi dan Peran Batch Inference

Pada penelitian ini, **batch inference berfungsi untuk meningkatkan efisiensi komputasi inferensi CNN** dengan memproses beberapa citra secara simultan dalam satu kali eksekusi model.

Keuntungan penggunaan batch inference meliputi:

1. **Mengurangi execution time**

   Model tidak perlu dipanggil berulang kali untuk setiap gambar.

2. **Meningkatkan throughput**

   Lebih banyak gambar dapat diproses setiap detik.

3. **Memanfaatkan optimisasi paralel TensorFlow**

   Operasi tensor dapat dijalankan secara paralel pada CPU maupun GPU.

4. **Mengurangi overhead inferensi**

   Alokasi memori dan eksekusi kernel dilakukan sekali untuk satu batch besar.

Berdasarkan hasil eksperimen penelitian, penggunaan batch inference menghasilkan penurunan waktu komputasi serta peningkatan throughput dibandingkan metode serial, terutama pada dataset berukuran besar.
