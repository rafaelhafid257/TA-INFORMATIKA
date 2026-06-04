# PANDUAN KONTEKS & INSTRUKSI PENGERJAAN PROYEK (ANTI-HALUSINASI)

Dokumen ini berisi seluruh spesifikasi, batasan, dan logika teknis untuk proyek Tugas Akhir saya. **PENTING:** Jangan pernah berasumsi, menambah, atau mengubah parameter di luar data yang tertulis di bawah ini. Jika ada informasi yang tidak diketahui, tanyakan kepada saya sebelum memberikan jawaban.

---

## 1. Identitas Proyek & Domain Bisnis
* **Topik Utama:** Informatika Pariwisata (Tourism Informatics) & Data Science.
* **Judul Penelitian:** Klasifikasi Tingkat Kepopuleran Objek Wisata di Bali Menggunakan Algoritma SVM Berdasarkan Kategori dan Lokasi Geografis.
* **Tujuan Akhir:** Membangun *Decision Support System* untuk investor infrastruktur pariwisata dan pemerintah daerah berbasis data spasial digital.
* **Integrasi Sistem:** Model yang dilatih di Python (.ipynb) diekspor ke format `.joblib` untuk nantinya diintegrasikan ke sistem informasi berbasis Web (menggunakan Laravel).

---

## 2. Spesifikasi Dataset & Fitur (Wajib Patuh)
Dataset yang digunakan adalah data objek wisata di Bali dengan total **751 baris data bersih** setelah proses *Data Cleaning*.

### Fitur Input (Tepat 5 Atribut - Tidak Boleh Kurang atau Lebih):
1. **kategori_encoded (Kategorikal/Nominal):** Jenis wisata (Alam, Budaya, Religi, Kuliner, dll) yang ditransformasikan menggunakan *Label Encoder*.
2. **kabupaten_encoded (Kategorikal/Nominal):** Wilayah administratif di Bali (Gianyar, Badung, Buleleng, dll) yang ditransformasikan menggunakan *Label Encoder*.
3. **latitude (Numerik Geospasial):** Titik koordinat garis lintang (Format desimal sekitar -8.x).
4. **longitude (Numerik Geospasial):** Titik koordinat garis bujur (Format desimal sekitar 115.x).
5. **DistanceToCenter (Numerik Geospasial/Fitur Rekayasa):** Atribut ke-5 hasil *Feature Engineering*. Dihitung menggunakan **Rumus Haversine** untuk mengukur jarak geometris (dalam kilometer) dari koordinat tempat wisata ke titik pusat kota Denpasar (Titik referensi Lapangan Puputan: `-8.6586, 115.2106`).

### Atribut yang Dibuang/Tidak Dipakai:
* **Nama tempat & Link URL:** Dibuang saat *Feature Selection* karena memiliki *high cardinality* dan tidak relevan secara statistik untuk proses generalisasi algoritma.

---

## 3. Logika Target & Klasifikasi (Multi-class)
Proyek ini menggunakan pendekatan **Multi-class Classification** yang membagi data ke dalam **3 kelas target** berdasarkan kolom `rating` asli di Google Maps:
* **Kelas 2 (Sempurna):** Objek wisata dengan rating tepat **5.0** (Total: 44 data).
* **Kelas 1 (Bagus/Recommended):** Objek wisata dengan rating **4.5 s.d 4.9** (Total: 465 data).
* **Kelas 0 (Kurang/Average):** Objek wisata dengan rating **di bawah 4.5** (Total: 242 data).

---

## 4. Konfigurasi Teknis Algoritma (Pipeline ML)
Setiap kali menuliskan kode, analisis, atau penjelasan teori, ikuti aturan arsitektur berikut:
1. **Data Preprocessing:** Wajib menyertakan `StandardScaler` (Z-score) setelah encoding karena SVM berbasis jarak (*distance-based*) dan sangat sensitif terhadap perbedaan rentang nilai (terutama antara koordinat desimal dan jarak kilometer).
2. **Data Splitting:** Rasio data latih dan data uji adalah **80% : 20%**.
3. **Algoritma Utama:** `SVC` (Support Vector Classification) dari library `sklearn.svm`.
4. **Fungsi Kernel:** Wajib menggunakan **Kernel RBF (Radial Basis Function)** untuk menangani pola spasial koordinat yang non-linear.
5. **Hyperparameter Tuning:** Menggunakan `GridSearchCV` untuk mencari parameter `C` dan `gamma` yang paling optimal secara otomatis.
6. **Penanganan Imbalanced Data:** Karena jumlah Kelas 2 (Sempurna) sangat sedikit, wajib mengaktifkan parameter `class_weight='balanced'` pada inisialisasi model SVM.
7. **Metrik Evaluasi:** Menggunakan *Confusion Matrix* (Heatmap) dan *Classification Report* (mengukur Accuracy, Precision, Recall, dan F1-Score per kelas).

---

## 5. Perintah Pengendalian AI (*Guardrails*)
* **JANGAN** menyarankan algoritma lain seperti Random Forest, Naive Bayes, KNN, atau Deep Learning, kecuali saya memintanya secara eksplisit. Fokus riset ini adalah **SVM**.
* **JANGAN** mengubah pembagian kelas menjadi 2 kelas (Binary) atau 4 kelas. Tetap di **3 kelas** (Sempurna, Bagus, Kurang).
* **JANGAN** melakukan scraping ulang data baru atau menyarankan penambahan atribut eksternal di luar 5 atribut yang sudah disepakati di atas.
* **SELALU** hubungkan aspek informatika (pemrograman, matematika algoritma) dengan aspek pariwisata (potensi wilayah, investasi, kebijakan publik daerah) di setiap output analisis.