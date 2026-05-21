## Laporan Singkat Hasil Evaluasi Performa Komputasi

Penelitian ini melakukan evaluasi performa komputasi pada sistem klasifikasi penyakit daun jagung berbasis **Convolutional Neural Network (CNN)** dengan membandingkan dua metode inferensi, yaitu **serial processing** dan **parallel processing menggunakan batch inference**. Pengujian dilakukan menggunakan empat variasi ukuran dataset, yaitu sekitar **10, 50, 100, dan 500 gambar**.

Evaluasi menggunakan beberapa metrik, yaitu **execution time**, **speedup**, **throughput**, dan **consistency check**.

### Hasil Execution Time dan Speedup

Hasil pengujian menunjukkan bahwa metode paralel secara konsisten menghasilkan waktu komputasi yang lebih cepat dibandingkan metode serial pada seluruh skenario dataset.

| Dataset    | Serial (s) | Parallel (s) | Speedup |
| ---------- | ---------- | ------------ | ------- |
| 10 gambar  | 2,19       | 1,60         | 1,37x   |
| 50 gambar  | 4,42       | 3,01         | 1,47x   |
| 100 gambar | 8,66       | 3,86         | 2,24x   |
| 500 gambar | 44,50      | 11,66        | 3,82x   |

Pada dataset kecil sebanyak **13 gambar**, metode paralel memperoleh waktu komputasi sebesar **1,60 detik**, lebih cepat dibandingkan metode serial sebesar **2,19 detik**, menghasilkan _speedup_ sebesar **1,37 kali**.

Pada dataset **51 gambar**, metode paralel menghasilkan _speedup_ sebesar **1,47 kali**. Peningkatan performa mulai terlihat lebih jelas ketika jumlah data meningkat.

Performa paling signifikan diperoleh pada dataset **500 gambar**, di mana metode paralel mampu menyelesaikan inferensi dalam **11,66 detik**, sedangkan metode serial membutuhkan **44,50 detik**. Nilai _speedup_ sebesar **3,82 kali** menunjukkan bahwa pendekatan batch inference memberikan keuntungan komputasi yang semakin besar pada dataset berukuran besar.

Hasil ini menunjukkan bahwa peningkatan jumlah data menyebabkan metode serial mengalami kenaikan waktu komputasi yang cukup tinggi, sementara metode paralel mampu mempertahankan pertumbuhan waktu komputasi yang lebih rendah.

### Hasil Throughput

Evaluasi throughput dilakukan untuk mengetahui jumlah gambar yang dapat diproses sistem setiap detik.

| Dataset    | Serial Throughput | Parallel Throughput |
| ---------- | ----------------- | ------------------- |
| 13 gambar  | 5,92 img/s        | 8,11 img/s          |
| 51 gambar  | 11,53 img/s       | 16,97 img/s         |
| 100 gambar | 11,54 img/s       | 25,87 img/s         |
| 500 gambar | 11,23 img/s       | 42,87 img/s         |

Berdasarkan hasil pengujian, metode paralel menunjukkan throughput yang lebih tinggi dibandingkan serial pada seluruh skenario pengujian.

Pada dataset **500 gambar**, throughput metode serial mencapai **11,23 gambar per detik**, sedangkan metode paralel mampu mencapai **42,87 gambar per detik**. Hal ini menunjukkan bahwa metode batch inference mampu meningkatkan kapasitas pemrosesan model CNN secara signifikan.

### Hasil Consistency Check

Selain aspek performa komputasi, penelitian ini juga melakukan pengujian **consistency check** untuk memastikan bahwa optimisasi paralel tidak mengubah hasil klasifikasi model.

Hasil pengujian menunjukkan bahwa seluruh dataset menghasilkan nilai **True** pada consistency check. Hal ini berarti prediksi yang dihasilkan metode paralel identik dengan metode serial.

Dengan demikian, penggunaan batch parallel inference mampu meningkatkan performa komputasi tanpa menyebabkan perubahan pada hasil prediksi model CNN.

### Kesimpulan Singkat

Secara keseluruhan, implementasi **parallel processing berbasis batch inference** terbukti lebih efisien dibandingkan serial processing pada klasifikasi penyakit daun jagung menggunakan CNN. Keunggulan metode paralel menjadi semakin terlihat seiring bertambahnya ukuran dataset, yang ditunjukkan oleh penurunan execution time, peningkatan throughput, dan nilai speedup yang lebih tinggi. Selain itu, consistency check menunjukkan bahwa peningkatan performa tersebut tidak mengorbankan konsistensi hasil klasifikasi model.
