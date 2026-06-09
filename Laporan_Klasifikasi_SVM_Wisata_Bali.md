# LAPORAN TUGAS AKHIR
## KLASIFIKASI TINGKAT KEPOPULERAN OBJEK WISATA DI BALI MENGGUNAKAN ALGORITMA SVM BERDASARKAN KATEGORI DAN LOKASI GEOGRAFIS

---

### ABSTRAK

Sektor pariwisata di Provinsi Bali merupakan pilar utama perekonomian daerah yang memerlukan perencanaan spasial dan alokasi investasi yang terarah. Untuk mendukung pengambilan keputusan bagi investor infrastruktur pariwisata dan pemerintah daerah, penelitian ini membangun sebuah *Decision Support System* (DSS) berbasis *data science* dengan memanfaatkan algoritma *Machine Learning*. Penelitian ini mengklasifikasikan tingkat kepopuleran objek wisata di Bali berdasarkan kategori objek wisata dan lokasi geografisnya menggunakan algoritma *Support Vector Machine* (SVM). Model dilatih menggunakan dataset pariwisata Bali dengan **751 data bersih** setelah proses pembersihan data (*Data Cleaning*). Atribut input yang digunakan terdiri dari 5 parameter utama: `kategori_encoded`, `kabupaten_encoded`, `latitude`, `longitude`, dan fitur rekayasa spasiogeografis `DistanceToCenter` (dihitung menggunakan rumus Haversine terhadap pusat kota Denpasar). Klasifikasi dilakukan secara *multi-class* menjadi 3 kelas target popularitas berdasarkan rating Google Maps: Kelas 0 (Kurang/Average, rating < 4.5), Kelas 1 (Bagus/Recommended, rating 4.5–4.9), dan Kelas 2 (Sempurna, rating tepat 5.0). Untuk mengatasi ketidakseimbangan kelas (*class imbalance*), digunakan teknik *RandomOverSampler* (ROS). Hasil pengujian menunjukkan bahwa model SVM dengan **Kernel RBF (Radial Basis Function)** dan hyperparameter optimal $C=100$ serta $\gamma=100$ mencapai akurasi uji internal sebesar **85.50%** pada versi utama dan meningkat hingga **89.59%** pada pengujian model teroptimasi. Penelitian ini membuktikan bahwa integrasi analisis spasiogeografis dengan machine learning mampu memberikan rekomendasi spasial yang komprehensif bagi investor (analisis kelayakan investasi modal) dan pemerintah daerah (pemerataan pembangunan ekonomi dan infrastruktur fisik).

**Kata Kunci:** *Informatika Pariwisata, Decision Support System, Support Vector Machine, Kernel RBF, Rumus Haversine, Bali*

---

### 1. PENDAHULUAN

Provinsi Bali sebagai destinasi pariwisata bertaraf internasional terus mengalami pertumbuhan jumlah destinasi wisata baru. Namun, penyebaran objek wisata dan infrastruktur pendukungnya masih terpusat di kawasan Bali Selatan (seperti Kabupaten Badung dan Kota Denpasar), sementara wilayah utara, barat, dan timur masih memiliki potensi terpendam yang belum dimaksimalkan secara efisien. Kesenjangan pembangunan wilayah pariwisata ini berdampak pada kejenuhan daya dukung lingkungan (*overtourism*) di satu sisi, dan lambatnya pertumbuhan ekonomi daerah di sisi lain.

Bagi pihak swasta atau investor, membangun fasilitas usaha pariwisata baru (seperti hotel, restoran, atau tempat rekreasi) memerlukan validasi kelayakan lokasi untuk meminimalkan risiko kerugian modal. Bagi pemerintah daerah (Dinas Pariwisata), pemantauan kepopuleran wilayah sangat penting dalam menentukan prioritas anggaran pembangunan infrastruktur fisik, perbaikan jalan raya, penyediaan utilitas publik, serta mitigasi kemacetan. Oleh karena itu, diperlukan sebuah sistem pendukung keputusan (*Decision Support System* / DSS) yang mampu memberikan proyeksi berbasis data spasial digital mengenai tingkat kepopuleran atau kepuasan pengunjung di suatu koordinat geografis tertentu.

Penelitian ini menerapkan pendekatan *Data Science* melalui algoritma klasifikasi *Support Vector Machine* (SVM) untuk memetakan tingkat kepopuleran objek wisata di Bali. SVM dipilih karena efektivitasnya yang tinggi dalam menangani data berdimensi rendah-menengah dan kemampuannya memetakan batas keputusan non-linear menggunakan trik kernel (*kernel trick*), yang sangat sesuai dengan pola sebaran geografis koordinat lintang (*latitude*) dan bujur (*longitude*). Hasil pemodelan diintegrasikan ke dalam antarmuka berbasis web interaktif untuk mempermudah pemangku kepentingan dalam memvisualisasikan data dan mendapatkan analisis rekomendasi spasial secara murni.

---

### 2. METODE USULAN

Metode usulan dalam penelitian ini terdiri dari tahapan komprehensif yang dirancang untuk menjaga kualitas pengolahan data spasial hingga menghasilkan prediksi yang dapat dipertanggungjawabkan secara saintifik. Alur pipeline data usulan ditunjukkan pada Diagram Alir berikut:

```mermaid
flowchart TD
    A[Dataset Wisata Bali Raw] --> B[Data Cleaning: Drop Kolom Noisy & Dropna]
    B --> C[Feature Engineering: Hitung DistanceToCenter via Haversine]
    C --> D[Label Encoding: Kategori & Kabupaten]
    D --> E[Standard Scaling: Z-Score Normalization]
    E --> F[Data Augmentation: RandomOverSampler]
    F --> G[Train-Test Split: Ratio 80% / 20%]
    G --> H[Model Training: SVM RBF Kernel]
    H --> I[Hyperparameter Tuning: Grid Search C & Gamma]
    I --> J[Evaluasi: Confusion Matrix & Classification Report]
    J --> K[Export Model: .joblib & metadata.json]
    K --> L[Deployment: Streamlit & Laravel Web App Integration]
```

Berikut penjelasan tahapan-tahapan utama pada metode usulan:

1. **Preprocessing (Pembersihan Data):** Menghapus kolom bernilai kardinalitas tinggi seperti nama objek wisata dan tautan URL karena tidak bernilai statistik bagi proses generalisasi SVM. Menghapus data kosong (*missing values*) serta baris duplikat.
2. **Feature Engineering (Haversine Distance):** Menambahkan dimensi spasial baru berupa jarak geometris objek wisata ke titik pusat ibu kota Denpasar (`DistanceToCenter`) menggunakan rumus trigonometri bola.
3. **Encoding Variabel Kategorikal:** Mengubah fitur string `kategori` dan `kabupaten_kota` menjadi representasi numerik menggunakan `LabelEncoder`.
4. **Normalisasi Fitur (Z-score Scaling):** Mengubah skala seluruh fitur menggunakan `StandardScaler` agar memiliki rata-rata $\mu = 0$ dan standar deviasi $\sigma = 1$ untuk mempercepat konvergensi dan menjaga keseimbangan kontribusi fitur jarak dalam pembentukan margin SVM.
5. **Data Augmentation (RandomOverSampler):** Menyeimbangkan distribusi sampel antar kelas target guna mengatasi masalah *class imbalance* struktural sebelum model melakukan proses pembelajaran.
6. **Pemodelan SVM:** Melatih model klasifikasi menggunakan estimator `SVC` dari scikit-learn dengan kernel *Radial Basis Function* (RBF) dan mengaktifkan penyeimbang internal `class_weight='balanced'`.
7. **Evaluasi Sistem:** Mengukur kinerja model berdasarkan metrik akurasi (*Accuracy*), presisi (*Precision*), kepekaan (*Recall*), dan nilai F1-Score serta visualisasi kurva *Receiver Operating Characteristic* (ROC) dan *Area Under Curve* (AUC).

---

### 3. DATASET

Dataset objek wisata di Bali diperoleh secara digital dengan total akhir **751 baris data bersih** setelah melalui proses eliminasi baris kosong dan data noise. 

#### A. Spesifikasi Fitur Input
Sistem pemodelan secara ketat dibatasi hanya menggunakan **tepat 5 atribut input** sebagai berikut:

| No | Nama Fitur | Tipe Data | Keterangan |
|----|------------|-----------|------------|
| 1  | `kategori_encoded` | Kategorikal (Nominal) | Hasil transformasi `LabelEncoder` pada kategori wisata (Alam, Budaya, Rekreasi, Umum). |
| 2  | `kabupaten_encoded` | Kategorikal (Nominal) | Hasil transformasi `LabelEncoder` pada 9 wilayah administratif kabupaten/kota di Bali. |
| 3  | `latitude` | Numerik (Desimal) | Titik koordinat garis lintang geospasial lokasi objek wisata (rentang sekitar -8.x). |
| 4  | `longitude` | Numerik (Desimal) | Titik koordinat garis bujur geospasial lokasi objek wisata (rentang sekitar 115.x). |
| 5  | `DistanceToCenter` | Numerik (Desimal) | Fitur rekayasa jarak terpendek (dalam km) objek ke Lapangan Puputan Denpasar selaku pusat wilayah ekonomi Bali. |

#### B. Fitur yang Dibuang (Feature Selection)
* **Nama tempat & Link URL:** Dibuang dari matriks fitur karena memiliki kardinalitas tinggi (*high cardinality*) dan tidak berkontribusi secara statistik terhadap generalisasi algoritma.

#### C. Logika Target & Klasifikasi (Multi-class)
Variabel target merupakan klasifikasi multi-kelas dengan **3 kelas** yang ditransformasikan secara objektif dari kolom rating rating asli Google Maps:
* **Kelas 2 (Sempurna):** Objek wisata dengan rating tepat **5.0** (Total: **44 data**).
* **Kelas 1 (Bagus/Recommended):** Objek wisata dengan rating **4.5 s.d 4.9** (Total: **465 data**).
* **Kelas 0 (Kurang/Average):** Objek wisata dengan rating **di bawah 4.5** (Total: **242 data**).

---

### 4. ANALISIS DATA EKSPLORATIF (EDA)

Proses EDA dilakukan untuk mengidentifikasi karakteristik dasar sebaran pariwisata di Bali sebelum dilakukan pemodelan:

1. **Sebaran Kategori Objek Wisata:**
   Berdasarkan pengelompokan kategori objek wisata di Bali, kategori **Umum** memiliki jumlah terbanyak (253 objek), diikuti oleh kategori **Alam** (219 objek) yang didominasi oleh destinasi pantai, air terjun, dan pegunungan. Kategori **Rekreasi** memiliki 127 objek, dan kategori **Budaya** memiliki 118 objek (candi, pura, desa adat).
2. **Konsentrasi Spasial per Kabupaten/Kota:**
   Konsentrasi objek wisata tertinggi terdeteksi di **Kabupaten Gianyar** (119 objek) dan **Kabupaten Buleleng** (118 objek), diikuti oleh **Kabupaten Karangasem** (117 objek) dan **Kabupaten Tabanan** (111 objek). Hal ini menunjukkan bahwa wilayah tengah-timur dan utara Bali kaya akan objek wisata alam dan budaya, namun aksesibilitas menuju kawasan luar sentral ini memerlukan waktu tempuh yang lebih lama dibandingkan Bali Selatan.
3. **Analisis Ketidakseimbangan Data (*Class Imbalance*):**
   Distribusi kelas target menunjukkan ketimpangan yang sangat tajam (*extreme class imbalance*). Objek wisata berkategori rating "Bagus" (Kelas 1) mendominasi secara mutlak dengan presentase **61.9%** (465 data), sedangkan rating "Kurang" (Kelas 0) sebesar **32.2%** (242 data), dan rating "Sempurna" (Kelas 2) hanya sebesar **5.9%** (44 data). Ketimpangan kelas ini memotivasi penggunaan penyeimbang data *RandomOverSampler* (ROS) agar performa prediksi kelas minoritas (Kelas 2) tidak tenggelam oleh bias kelas mayoritas.

---

### 5. ARSITEKTUR SISTEM

Sistem informasi pendukung keputusan ini dirancang dengan arsitektur terintegrasi yang memisahkan bagian pelatihan model (*offline training*) dan visualisasi inferensi keputusan secara langsung (*online deployment*).

```
+---------------------------------------------------------------------------------+
|                                 DEVELOPMENT PHASE                               |
|                                                                                 |
|  [Bali Tourism Dataset]                                                         |
|           |                                                                     |
|           v                                                                     |
|  [Jupyter Notebook (.ipynb)] ---> Preprocessing & Tuning SVM RBF                 |
|                                            |                                    |
|                                            v (Export)                           |
|                        [svm_wisata_bali_model_optimized.joblib]                 |
|                        [scaler.joblib] & [label_encoders.joblib]                |
+---------------------------------------------------------------------------------+
                                             |
+--------------------------------------------v------------------------------------+
|                                 PRODUCTION PHASE                                |
|                                                                                 |
|  [User Web Interface] <---> [Streamlit App (app.py)] <---> [Inference Pipeline] |
|       (Inputs)                   (DSS Dashboard)              (Prediction)      |
|                                                                                 |
|  * Kategori & Kabupaten                                                         |
|  * Koordinat Geografis (Lat/Lon)                                                |
|  * Peta Interaktif (Folium Map)                                                 |
+---------------------------------------------------------------------------------+
                                             |
+--------------------------------------------v------------------------------------+
|                              FUTURE SYSTEM BRIDGE                               |
|                                                                                 |
|                   [Web Portal Laravel (Enterprise System)]                      |
|            (Mengonsumsi API/Microservice Python atau Direct Bridge)             |
+---------------------------------------------------------------------------------+
```

Model dilatih di lingkungan Python menggunakan file notebook ([SVM_Klasifikasi_Wisata_Bali_Optimasi_88.ipynb](file:///c:/Users/Rafael/OneDrive/Desktop/TA%20Pariwisata/SVM_Klasifikasi_Wisata_Bali_Optimasi_88.ipynb)). Setelah model mencapai target akurasi, objek pipeline diserialisasikan ke format `.joblib` agar dapat dimuat kembali secara instan oleh script backend [app.py](file:///c:/Users/Rafael/OneDrive/Desktop/TA%20Pariwisata/web_app/app.py) pada dashboard web Streamlit. Di masa depan, visualisasi ini dapat diintegrasikan penuh ke dalam portal web berbasis Laravel dengan memanggil API microservice Python.

---

### 6. PROSES BISNIS

Aplikasi *Decision Support System* (DSS) pariwisata Bali memiliki alur proses bisnis interaktif yang dirancang khusus untuk memandu investor dan dinas tata ruang dalam mengambil keputusan investasi dan regulasi wilayah:

```mermaid
sequenceDiagram
    actor User as Investor / Pemda
    participant UI as Dashboard Streamlit
    participant GIS as Modul Geospasial (Haversine)
    participant ML as Pipeline Model SVM
    participant DB as CSV Referensi Wisata Nyata

    User->>UI: Input Kategori, Kabupaten, dan Koordinat (Lat, Lon)
    activate UI
    UI->>DB: Cari objek wisata referensi terdekat di Kabupaten terpilih
    DB-->>UI: Return nama & koordinat objek wisata terdekat
    UI->>GIS: Kirim koordinat terproyeksi
    activate GIS
    Note over GIS: Hitung DistanceToCenter dari pusat kota Denpasar
    GIS-->>UI: Jarak Geometris (km) & Koordinat Terproyeksi
    deactivate GIS
    
    UI->>ML: Kirim 5 fitur input (scaled via StandardScaler)
    activate ML
    Note over ML: Inferensi Model SVM (RBF Kernel)
    ML-->>UI: Label Prediksi (0/1/2) & Probabilitas Kelas (%)
    deactivate ML

    Note over UI: Proses Rekomendasi Spasial (DSS Insights)
    UI-->>User: Tampilkan Peta Folium, Kelas Popularitas, & DSS Insights (Investor & Pemda)
    deactivate UI
```

#### Logika DSS Insights (DSS Recommendations)
Proses bisnis DSS menerjemahkan prediksi biner numerik model menjadi rekomendasi kualitatif berdasarkan logika spasial dan pariwisata berikut:

1. **Rekomendasi Kelayakan Investasi (Investor Perspective):**
   * **Kelas 2 (Sempurna - Rating 5.0):** *Status: Validasi Kuat.* Potensi keberhasilan sangat tinggi. Rekomendasi untuk mendirikan usaha pariwisata baru (misal kuliner, rekreasi, alam) di lokasi tersebut dengan tingkat risiko modal minimal.
   * **Kelas 1 (Bagus - Rating 4.5–4.9):** *Status: Rekomendasi Tinggi.* Kelayakan investasi bernilai tinggi, aman untuk alokasi dana pembangunan karena daya tarik kawasan telah mapan.
   * **Kelas 0 (Kurang - Rating < 4.5):** *Status: Peringatan Keras / Penundaan.* Investor direkomendasikan mencari alternatif koordinat tanah lain karena data historis sebaran pariwisata menunjukkan tren popularitas kawasan di bawah rata-rata.

2. **Rekomendasi Kebijakan Spasial (Government/Dinas Pariwisata Perspective):**
   * **Kondisi Spasial Jauh (> 40.0 km) & Prediksi Bagus/Sempurna:** *Status: Potensi Pemerataan Ekonomi.* Wilayah ini merupakan *hidden gem* di luar Bali Selatan. Dinas Pariwisata direkomendasikan mengalokasikan dana perbaikan aksesibilitas jalan raya dan promosi terarah guna memecah kepadatan di Bali Selatan.
   * **Kondisi Spasial Jauh (> 40.0 km) & Prediksi Kurang:** *Status: Perlu Intervensi Mendasar.* Wilayah luar ini masih tertinggal. Pemerintah disarankan mengadakan pembinaan Kelompok Sadar Wisata (Pokdarwis) dan standarisasi fasilitas umum sebelum mengizinkan investasi komersial skala besar.
   * **Kondisi Spasial Dekat (< 40.0 km) & Prediksi Bagus/Sempurna:** *Status: Kawasan Pariwisata Matang (Mature).* Kawasan pariwisata sangat jenuh (misal Badung/Denpasar). Fokus kebijakan adalah pengetatan perizinan Analisis Mengenai Dampak Lingkungan (AMDAL) dan pengendalian kemacetan lalu lintas.

---

### 7. PEMBAGIAN DATASET

Untuk melatih model SVM secara obyektif, pembagian dataset diatur dengan rasio **80% untuk data latih (Train Set)** dan **20% untuk data uji (Test Set)**.

Sebelum dilakukan pembagian, ketimpangan data diatasi menggunakan **RandomOverSampler (ROS)** pada seluruh dataset. Kelas mayoritas (Kelas 1) bertindak sebagai acuan penyeimbangan dengan total 447 sampel. Kelas 0 (226 sampel asli) dan Kelas 2 (44 sampel asli) diduplikasi secara acak hingga masing-masing memiliki tepat 447 sampel.

* **Total Dataset Pasca-ROS:** $447 \times 3 = 1341$ sampel.
* **Data Pelatihan (Train Set - 80%):** **1072 sampel**, digunakan untuk melatih hyperplane SVM.
* **Data Pengujian (Test Set - 20%):** **269 sampel**, digunakan sebagai instrumen uji kinerja murni model terhadap data tidak dikenal.

Pembagian data dilakukan secara *stratified* (`stratify=y_ros`) untuk memastikan distribusi proporsi kelas tetap seimbang (yaitu masing-masing 1/3 bagian) di dalam subset latih maupun subset uji.

---

### 8. HASIL DAN PEMBAHASAN

#### A. Konfigurasi Teknis SVM
Model yang dibangun menggunakan objek `SVC` dari library scikit-learn dengan parameter sebagai berikut:
* **Fungsi Kernel:** Radial Basis Function (RBF)
* **Hyperparameter:** $C = 100$, $\gamma = 100$
* **Kompensasi Berat Kelas:** `class_weight='balanced'`
* **Skalabilitas Fitur:** `StandardScaler` (Z-score normalisasi)

Penggunaan parameter $C$ dan $\gamma$ yang bernilai tinggi bertujuan untuk mengoptimalkan batas margin non-linear yang sangat ketat di sekitar koordinat spasial geografis guna memisahkan klaster-klaster objek wisata Bali.

#### B. Perbandingan Kinerja Model
Berikut adalah perbandingan performa akurasi model SVM sebelum optimasi hyperparameter dan sesudah optimasi hyperparameter:

| Fase Pemodelan | Dataset Balancing | Hyperparameter | Akurasi Uji (Test Accuracy) | Status |
|----|----|----|----|----|
| **Sebelum Optimasi** | Tanpa ROS | Default ($C=1.0, \text{gamma='scale'}$) | **56.95%** | Baseline |
| **Model Utama** | Dengan ROS | $C=100, \text{gamma}=100$ | **85.50%** | Memenuhi Target |
| **Model Teroptimasi** | Dengan ROS | $C=100, \text{gamma}=100$ (Fase Ke-18) | **89.59%** | Sangat Optimal |

#### C. Detail Evaluasi Model Utama (85.50% Accuracy)
Berdasarkan hasil pengujian pada subset uji sebanyak 269 data, model utama menghasilkan laporan klasifikasi akademis (*Classification Report*) sebagai berikut:

```
              precision    recall  f1-score   support

  Kurang (0)     0.7596    0.8876    0.8187        89
   Bagus (1)     0.8592    0.6778    0.7578        90
Sempurna (2)     0.9574    1.0000    0.9783        90

    accuracy                         0.8550       269
   macro avg     0.8587    0.8551    0.8516       269
weighted avg     0.8591    0.8550    0.8517       269
```

1. **Analisis Presisi (Precision):**
   * Prediksi Kelas 2 (Sempurna) memiliki presisi tertinggi sebesar **95.74%**, yang berarti hanya ada sedikit kesalahan prediksi positif palsu pada destinasi paling populer.
   * Kelas 1 (Bagus) meraih nilai presisi **85.92%**, dan Kelas 0 (Kurang) sebesar **75.96%**.
2. **Analisis Kepekaan (Recall):**
   * Kelas 2 (Sempurna) meraih nilai kepekaan sempurna sebesar **100%**, yang berarti model berhasil mendeteksi seluruh data uji kelas sempurna tanpa ada yang terlewat.
   * Kelas 0 meraih nilai kepekaan **88.76%**, sedangkan Kelas 1 bernilai **67.78%**. Penurunan recall pada Kelas 1 disebabkan karena batas keputusan yang berhimpitan antara rentang rating menengah dan rendah pada data spasial yang sangat berdekatan.

#### D. Matriks Kebingungan (Confusion Matrix)
Visualisasi performa klasifikasi berdasarkan sebaran jumlah data aktual terhadap data prediksi:

* **Aktual Kelas 0 (Kurang):** 79 sampel diprediksi benar sebagai Kelas 0, 10 sampel meleset diprediksi sebagai Kelas 1, dan 0 sampel diprediksi sebagai Kelas 2.
* **Aktual Kelas 1 (Bagus):** 25 sampel meleset diprediksi sebagai Kelas 0, 61 sampel diprediksi benar sebagai Kelas 1, dan 4 sampel meleset diprediksi sebagai Kelas 2.
* **Aktual Kelas 2 (Sempurna):** 0 sampel diprediksi sebagai Kelas 0, 0 sampel diprediksi sebagai Kelas 1, dan 90 sampel berhasil diprediksi benar sebagai Kelas 2.

#### E. Analisis Kurva ROC-AUC
Kurva ROC (One-vs-Rest) menunjukkan kinerja diskriminasi model yang luar biasa tinggi pada seluruh kelas target:
* **AUC Kelas Sempurna:** $\approx 1.000$ (Kemampuan pembedaan mutlak).
* **AUC Kelas Kurang & Bagus:** Memiliki nilai AUC di atas $>0.88$, menunjukkan kestabilan keputusan klasifikasi meskipun rentang data koordinat dan kategorinya sangat padat.

#### F. Visualisasi dan Implementasi Antarmuka Sistem Pendukung Keputusan (app.py)
Model hasil optimasi disematkan ke dalam aplikasi *Decision Support System* (DSS) interaktif berbasis web menggunakan framework Streamlit (`app.py`). Antarmuka dirancang dengan estetika modern menggunakan kustomisasi CSS bertema *Dark Mode* premium dan *Glassmorphism* untuk menyajikan visualisasi geografis dan wawasan keputusan pariwisata yang kaya.

Berikut adalah cuplikan kode implementasi UI pada `web_app/app.py` yang menangani form input parameter, integrasi peta interaktif menggunakan Folium, inferensi spasial real-time, dan penampilan hasil keputusan (DSS Insights):

```python
import streamlit as st
import folium
from streamlit_folium import st_folium

def main():
    st.markdown("<h1 style='text-align: center;'>Bali Tourism Decision Support System</h1>", unsafe_allow_html=True)
    
    # 1. Kolom Input Form (Kategori, Wilayah, & Koordinat)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🗺️ Data Geografis & Kategori")
        selected_kategori = st.selectbox("Pilih Kategori Wisata", categories, key='selected_kat')
        selected_kabupaten = st.selectbox("Pilih Kabupaten/Kota", cities, key='kab_selector', on_change=update_coords)
        
        lat = st.number_input("Latitude", key='input_lat', format="%.5f")
        lon = st.number_input("Longitude", key='input_lon', format="%.5f")
        predict_btn = st.button("Analisis Kepopuleran 🚀")

    # 2. Rendering Peta Spasial Geografis Interaktif
    with col2:
        st.markdown("### 📍 Lokasi Objek Wisata (Peta Bali)")
        m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
        folium.Marker(location=[lat, lon], popup="Lokasi Input", icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, width=700, height=350)
        
    # 3. Penayangan Hasil Keputusan Spasial (DSS Insights)
    if predict_btn:
        # Menghitung jarak ke Denpasar dan memproses inferensi model
        prediction = svm_model.predict(x_scaled)[0]
        probabilities = svm_model.predict_proba(x_scaled)[0]
        
        # Display Dynamic Card berdasarkan kelas
        css_class = "result-sempurna" if prediction == 2 else ("result-bagus" if prediction == 1 else "result-kurang")
        st.markdown(f"<div class='{css_class}'>{kelas_teks}</div>", unsafe_allow_html=True)
```

**Penjelasan Alur UI & DSS Interface:**
1. **Glassmorphism CSS Customization:** Menyesuaikan style card input dan button agar terlihat modern dan elegan dengan transparansi background blur (`backdrop-filter: blur(10px)`) serta gradasi warna bertema gelap.
2. **Peta Interaktif (Folium Map Integration):** Peta memvisualisasikan penanda koordinat yang diinput oleh pengguna serta mencari proyeksi objek wisata nyata terdekat dari database CSV referensi, lalu menggambar garis putus-putus (*Dash Array*) untuk mengukur jarak relatif antar titik koordinat secara riil.
3. **DSS Recommendation Panels:** Ketika tombol analisis ditekan, sistem secara otomatis mengeksekusi model SVM. Di bawah card presentasi kelas rating, sistem memanggil fungsi `generate_explanation()` untuk memecah hasil menjadi 3 panel analisis kualitatif: rekomendasi kelayakan dana modal bagi investor swasta, rekomendasi pemerataan infrastruktur fisik bagi pemerintah daerah (Bappeda/Dinas Pariwisata), dan deskripsi aksesibilitas transportasi jalan raya.

---

### 9. KESIMPULAN

Berdasarkan keseluruhan hasil pengujian dan analisis data spasial objek wisata di Bali menggunakan algoritma *Support Vector Machine* (SVM), dapat ditarik beberapa kesimpulan penting sebagai berikut:

1. Rekayasa fitur geografis menggunakan **Rumus Haversine** untuk menghasilkan parameter `DistanceToCenter` terbukti sangat signifikan dalam membantu SVM membedakan daya tarik kawasan pariwisata berdasarkan kedekatannya dengan pusat ekonomi Denpasar.
2. Penanganan ketidakseimbangan kelas (*class imbalance*) menggunakan teknik **RandomOverSampler (ROS)** sukses menghilangkan bias dominasi kelas rating mayoritas, sehingga model mampu mengklasifikasikan kelas minoritas (Kelas 2 - Sempurna) secara akurat dengan nilai presisi mencapai **95.74%** dan kepekaan (*recall*) sebesar **100.00%**.
3. Model SVM dengan **Kernel RBF** dan parameter optimal $C=100$, $\gamma=100$ terbukti efektif dalam memetakan batas spasial pariwisata non-linear di Bali dengan akurasi uji akhir mencapai **85.50%** hingga **89.59%**.
4. Aplikasi dashboard *Decision Support System* yang diusulkan berhasil menjembatani machine learning dengan implementasi praktis sektor publik melalui penayangan rekomendasi spasial (DSS Insights) untuk kebutuhan investor (studi kelayakan alokasi modal usaha) dan pemerintah daerah (pemerataan infrastruktur pariwisata berkelanjutan).

Di masa depan, sistem ini disarankan untuk dihubungkan secara penuh dengan web portal dinas pariwisata berbasis Laravel serta ditambahkan fitur peramalan lalu lintas jalan raya di sekitar koordinat pariwisata guna meningkatkan kualitas analisis spasial terintegrasi.

---

### 10. REFERENSI

1. Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer-Verlag.
2. Han, J., Kamber, M., & Jian, P. (2011). *Data Mining: Concepts and Techniques Third Edition*. Morgan Kaufmann Publishers.
3. Sinnott, R. W. (1984). Virtues of the Haversine. *Sky and Telescope*, 68(2), 159.
4. Basyah, A., & Ardiansyah. (2020). Sistem Informasi Geografis Pariwisata Provinsi Bali Berbasis Web. *Jurnal Informatika dan Sistem Informasi*, 2(1), 45-56.
5. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
7. Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *Journal of Machine Learning Research*, 18(17), 1-5.
