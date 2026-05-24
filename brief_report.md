## Laporan Singkat Hasil Evaluasi Performa Komputasi

### Spesifikasi Perangkat Sistem

Pengujian benchmark dilakukan pada lingkungan perangkat keras dan perangkat lunak dengan spesifikasi sebagai berikut:

- **CPU:** 12th Gen Intel(R) Core(TM) i5-12450H (Base Speed 2.00 GHz, 8 Cores: 4 P-cores + 4 E-cores)
- **RAM:** 24.576 MB (24 GB)
- **Versi Python:** 3.13.5
- **Sistem Operasi:** Windows 11 Home Single Language 64-Bit

---

Penelitian ini melakukan evaluasi performa komputasi pada sistem klasifikasi penyakit daun jagung berbasis **Convolutional Neural Network (CNN)** dengan membandingkan dua metode inferensi, yaitu **serial processing** dan **parallel processing menggunakan batch inference**. Pengujian dilakukan menggunakan tiga variasi ukuran dataset berskala besar, yaitu **1000, 2000, dan 3000 gambar**.

Evaluasi menggunakan beberapa metrik, yaitu **execution time**, **speedup**, **throughput**, dan **consistency check**.

### Hasil Execution Time dan Speedup

Hasil pengujian menunjukkan bahwa metode paralel secara konsisten menghasilkan waktu komputasi yang jauh lebih cepat dibandingkan metode serial pada seluruh skenario dataset berskala besar.

| Dataset     | Serial (s) | Parallel (s) | Speedup |
| ----------- | ---------- | ------------ | ------- |
| 1000 gambar | 72,84      | 16,08        | 4,53x   |
| 2000 gambar | 226,31     | 39,62        | 5,71x   |
| 3000 gambar | 216,18     | 42,97        | 5,03x   |

Pada dataset sebanyak **1000 gambar**, metode paralel memperoleh waktu komputasi sebesar **16,08 detik**, jauh lebih cepat dibandingkan metode serial sebesar **72,84 detik**, menghasilkan _speedup_ yang signifikan sebesar **4,53 kali**.

Performa paling optimal dan tertinggi diperoleh pada dataset **2000 gambar** (dengan total beban pemrosesan **3000 gambar**), di mana metode paralel menghasilkan _speedup_ sebesar **5,71 kali**. Metode paralel mampu menyelesaikan komputasi dalam waktu **39,62 detik**, sementara metode serial membutuhkan waktu hingga **226,31 detik**. Peningkatan performa ini menunjukkan bahwa pendekatan batch inference memberikan keuntungan efisiensi komputasi yang luar biasa ketika menangani ribuan gambar sekaligus memanfaatkan arsitektur _hybrid cores_ (P-cores & E-cores) pada prosesor Intel Core i5-12450H secara optimal.

Pada dataset **3000 gambar**, metode paralel mampu menyelesaikan inferensi dalam **42,97 detik**, sedangkan metode serial membutuhkan **216,18 detik**, menghasilkan nilai _speedup_ sebesar **5,03 kali**.

Hasil ini menegaskan bahwa peningkatan jumlah data menyebabkan metode serial mengalami lonjakan waktu komputasi yang sangat tinggi secara linier, sementara metode paralel mampu memanfaatkan pemrosesan multi-threading/batching secara efisien guna menekan pertumbuhan waktu komputasi tetap rendah.

### Hasil Throughput

Evaluasi throughput dilakukan untuk mengetahui jumlah gambar yang dapat diproses oleh sistem setiap detiknya.

| Dataset     | Serial Throughput | Parallel Throughput |
| ----------- | ----------------- | ------------------- |
| 1000 gambar | 13,73 img/s       | 62,17 img/s         |
| 2000 gambar | 13,26 img/s       | 75,71 img/s         |
| 3000 gambar | 13,88 img/s       | 69,82 img/s         |

Berdasarkan hasil pengujian, metode paralel menunjukkan kapasitas throughput yang berkali-kali lipat lebih tinggi dibandingkan metode serial pada seluruh skenario pengujian. Sementara kapasitas pemrosesan metode serial cenderung stagnan di kisaran **13 gambar per detik** tanpa memedulikan ukuran dataset, metode paralel mampu menembus angka pemrosesan yang jauh lebih masif.

Pada dataset **2000 gambar**, throughput metode serial hanya mencapai **13,26 gambar per detik**, sedangkan metode paralel mampu melesat hingga mencapai **75,71 gambar per detik**. Hal ini membuktikan secara empiris bahwa metode batch inference berhasil membuka keterbatasan perangkat keras dalam meningkatkan kapasitas pemrosesan model CNN secara optimal.

### Hasil Consistency Check

Selain aspek performa komputasi, penelitian ini juga melakukan pengujian **consistency check** untuk memastikan bahwa pengaktifan optimisasi paralel tidak mengubah atau merusak akurasi hasil klasifikasi model.

Hasil pengujian menunjukkan bahwa seluruh variasi pengujian dataset menghasilkan nilai **True** pada consistency check. Hal ini berarti prediksi yang dihasilkan oleh metode paralel 100% identik dan konsisten dengan metode serial.

Dengan demikian, penggunaan batch parallel inference terbukti aman karena mampu meningkatkan performa komputasi secara drastis tanpa menyebabkan perubahan atau penurunan kualitas pada hasil prediksi model CNN.

### Kesimpulan Singkat

Secara keseluruhan, implementasi **parallel processing berbasis batch inference** terbukti jauh lebih unggul dan efisien dibandingkan serial processing dalam menangani klasifikasi penyakit daun jagung menggunakan CNN pada skala ribuan data. Keunggulan metode paralel divalidasi oleh penurunan execution time yang drastis, peningkatan throughput hingga lebih dari 5x lipat, serta perolehan angka speedup yang tinggi pada spesifikasi sistem Intel Core i5 generasi ke-12. Selain itu, pengujian consistency check memberikan jaminan penuh bahwa peningkatan performa komputasi tersebut sama sekali tidak mengorbankan konsistensi dan akurasi hasil akhir prediksi model.
