import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(
    page_title="Bali Tourism DSS",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS (Glassmorphism & Premium Dark Mode) ---
def load_custom_css():
    st.markdown("""
    <style>
        /* Base Theme Override */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
        }
        
        [data-testid="stHeader"] {
            background: transparent;
        }

        /* Glassmorphism Cards */
        div.st-emotion-cache-1r6slb0.e1f1d6gn2, .glass-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }

        /* Input Fields */
        div[data-baseweb="select"] > div, input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 8px !important;
        }

        /* Button Styling */
        button[data-testid="baseButton-secondary"] {
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 10px 25px !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        button[data-testid="baseButton-secondary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(139, 92, 246, 0.4) !important;
        }

        /* Typography */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        p, label {
            font-family: 'Inter', sans-serif;
            color: #cbd5e1;
        }
        
        /* Result Alerts */
        .result-sempurna {
            padding: 20px;
            border-radius: 12px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid #10b981;
            color: #34d399;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        .result-bagus {
            padding: 20px;
            border-radius: 12px;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid #3b82f6;
            color: #60a5fa;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        .result-kurang {
            padding: 20px;
            border-radius: 12px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #f87171;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

# --- Backend Processing ---
DENPASAR_LAT = -8.6586
DENPASAR_LON = 115.2106

# Titik Koordinat Pusat Default Setiap Kabupaten (Kasar)
KABUPATEN_COORDS = {
    'Badung': (-8.5814, 115.1772),
    'Bangli': (-8.2574, 115.3533),
    'Buleleng': (-8.1120, 115.0889),
    'Denpasar': (-8.6500, 115.2167),
    'Gianyar': (-8.4326, 115.2917),
    'Jembrana': (-8.3361, 114.6190),
    'Karangasem': (-8.3844, 115.5406),
    'Klungkung': (-8.5333, 115.4167),
    'Tabanan': (-8.4419, 115.1146)
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_resource
def load_dependencies():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'references', 'dataset_tempat_wisata_bali(1).csv')
    model_path = os.path.join(base_dir, 'svm_model_files', 'svm_wisata_bali_model_optimized.joblib')
    
    svm_model = joblib.load(model_path)
    
    df = pd.read_csv(dataset_path)
    
    df_ref = df[['nama', 'kategori', 'kabupaten_kota', 'latitude', 'longitude']].copy()
    df_ref['latitude'] = pd.to_numeric(df_ref['latitude'], errors='coerce')
    df_ref['longitude'] = pd.to_numeric(df_ref['longitude'], errors='coerce')
    df_ref = df_ref.dropna()
    
    # Filter sensor kata-kata konyol / troll di Google Maps agar aman saat sidang TA
    troll_pattern = r'berak|kencing|kontol|memek|jembut|ngentot|peli|toket|troll|sampah|berak masal|toilet'
    df_ref = df_ref[~df_ref['nama'].str.lower().str.contains(troll_pattern, na=False)]
    
    df_clean = df.drop(columns=['nama', 'preferensi', 'link'])
    df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
    df_clean = df_clean.dropna()
    
    df_clean['DistanceToCenter'] = df_clean.apply(
        lambda row: haversine_distance(DENPAR_LAT := DENPASAR_LAT, DENPASAR_LON, row['latitude'], row['longitude']),
        axis=1
    )
    
    le_kategori = LabelEncoder()
    le_kabupaten = LabelEncoder()
    df_clean['kategori_encoded'] = le_kategori.fit_transform(df_clean['kategori'])
    df_clean['kabupaten_encoded'] = le_kabupaten.fit_transform(df_clean['kabupaten_kota'])
    
    feature_columns = ['kategori_encoded', 'kabupaten_encoded', 'latitude', 'longitude', 'DistanceToCenter']
    X = df_clean[feature_columns]
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    categories = sorted(df_clean['kategori'].unique().tolist())
    cities = sorted(df_clean['kabupaten_kota'].unique().tolist())
    
    return svm_model, le_kategori, le_kabupaten, scaler, categories, cities, df_ref

def find_nearest_spot(target_lat, target_lon, target_kabupaten, df_ref):
    """Mencari objek wisata nyata terdekat dari kabupaten terpilih"""
    df_kab = df_ref[df_ref['kabupaten_kota'].str.strip().str.lower() == target_kabupaten.strip().lower()]
    if df_kab.empty:
        df_kab = df_ref
        
    distances = df_kab.apply(
        lambda row: haversine_distance(target_lat, target_lon, row['latitude'], row['longitude']),
        axis=1
    )
    
    idx_min = distances.idxmin()
    nearest_row = df_kab.loc[idx_min]
    min_dist = distances.loc[idx_min]
    
    return {
        'nama': nearest_row['nama'],
        'kategori': nearest_row['kategori'],
        'latitude': nearest_row['latitude'],
        'longitude': nearest_row['longitude'],
        'distance_km': min_dist
    }

def generate_explanation(prediction, distance, category, kabupaten):
    """Menghasilkan penjelasan kualitatif pariwisata dan wawasan spasial (DSS Insights) untuk Investor & Pemerintah"""
    
    # 1. Perspektif Kelayakan Investasi (Investor Perspective)
    if prediction == 2:
        investor_insight = (
            "🏦 **Rekomendasi Kelayakan Investasi (Investor):**<br>"
            "<b>STATUS: VALIDASI KUAT (Sempurna - Rating 5.0)</b><br>"
            "Lokasi ini memiliki potensi keberhasilan sangat tinggi. Sangat direkomendasikan untuk mencairkan anggaran pembangunan "
            f"usaha baru pada kategori <b>{category}</b>. Risiko kerugian modal dinilai minimal karena daya dukung wilayah "
            "dan ulasan historis menunjukkan tingkat apresiasi pengunjung mutlak."
        )
    elif prediction == 1:
        investor_insight = (
            "🏦 **Rekomendasi Kelayakan Investasi (Investor):**<br>"
            "<b>STATUS: REKOMENDASI TINGGI (Bagus - Rating 4.5 - 4.9)</b><br>"
            "Lokasi ini sangat layak untuk dikembangkan. Investor memiliki dasar ilmiah yang kuat untuk memulai proyek fisik. "
            f"Pengembangan usaha kategori <b>{category}</b> di wilayah ini berpotensi mendatangkan profit yang stabil "
            "karena berada pada klaster pariwisata dengan minat wisatawan yang mapan."
        )
    else:
        investor_insight = (
            "🏦 **Rekomendasi Kelayakan Investasi (Investor):**<br>"
            "<b>STATUS: PERINGATAN KERAS / PENUNDAAN (Kurang - Rating < 4.5)</b><br>"
            "Investor disarankan <b>menunda pencairan modal</b> atau mencari alternatif koordinat lahan lain. "
            f"Melanjutkan investasi usaha kategori <b>{category}</b> di titik ini memiliki risiko kerugian finansial yang signifikan "
            "karena data historis menunjukkan tingkat popularitas atau kepuasan pengunjung di wilayah sekitar berada di bawah rata-rata."
        )

    # 2. Perspektif Kebijakan Wilayah & Pemerataan (Pemerintah/Dinas Pariwisata)
    if distance > 40.0:
        if prediction in [1, 2]:
            gov_insight = (
                "🏛️ **Rekomendasi Kebijakan Spasial (Pemerintah):**<br>"
                f"<b>STATUS: POTENSI PEMERATAAN EKONOMI (Hidden Gem di {kabupaten})</b><br>"
                f"Lokasi ini berada jauh dari pusat pariwisata Bali Selatan ({distance:.2f} km) namun diprediksi memiliki popularitas tinggi. "
                "Dinas Pariwisata direkomendasikan menjadikan wilayah ini sebagai prioritas alokasi dana perbaikan jalan, penerangan jalan, "
                "dan promosi pariwisata digital guna memecah kepadatan wisatawan di Bali Selatan."
            )
        else:
            gov_insight = (
                "🏛️ **Rekomendasi Kebijakan Spasial (Pemerintah):**<br>"
                f"<b>STATUS: PERLU INTERVENSI MENDASAR (Kawasan Tertinggal)</b><br>"
                f"Wilayah di luar pusat pariwisata ini ({distance:.2f} km) masih memerlukan perbaikan mendasar. "
                "Pemerintah disarankan memprioritaskan pelatihan kelompok sadar wisata (Pokdarwis) serta standardisasi sanitasi dan fasilitas "
                "kebersihan umum sebelum mengizinkan investasi komersial skala besar."
            )
    else:
        if prediction in [1, 2]:
            gov_insight = (
                "🏛️ **Rekomendasi Kebijakan Spasial (Pemerintah):**<br>"
                f"<b>STATUS: KAWASAN PARIWISATA MATANG (Jenuh/Mature Cluster)</b><br>"
                f"Kawasan ini sangat dekat dengan pusat pariwisata ({distance:.2f} km). "
                "Fokus pemerintah disarankan bukan lagi pada promosi tambahan atau alokasi dana pemerataan, "
                "melainkan pada pengetatan regulasi tata ruang (AMDAL) dan pengendalian kemacetan akibat kepadatan usaha."
            )
        else:
            gov_insight = (
                "🏛️ **Rekomendasi Kebijakan Spasial (Pemerintah):**<br>"
                f"<b>STATUS: OPTIMASI KAWASAN SENTRAL (Pusat Hub)</b><br>"
                f"Meskipun dekat dengan pusat pariwisata ({distance:.2f} km), titik ini diprediksi kurang populer. "
                "Dinas Pariwisata perlu melakukan audit kelayakan fasilitas umum sekitar dan merevitalisasi infrastruktur pendukung "
                "agar kawasan ini dapat kembali kompetitif."
            )

    # 3. Analisis Aksesibilitas Geografis (Haversine Distance Insight)
    if distance < 15.0:
        spatial_desc = (
            f"📍 **Aksesibilitas Geografis:** Lokasi berjarak {distance:.2f} km dari Denpasar (Sangat Dekat). "
            "Akses yang dekat dengan bandara dan pusat akomodasi memberikan keuntungan aksesibilitas tinggi bagi wisatawan."
        )
    elif distance < 40.0:
        spatial_desc = (
            f"📍 **Aksesibilitas Geografis:** Jarak ke Denpasar adalah {distance:.2f} km (Sedang). "
            "Lokasi ini berada dalam radius perjalanan harian (*day trip*) yang ideal bagi wisatawan yang menginap di kawasan Bali Selatan."
        )
    else:
        spatial_desc = (
            f"📍 **Aksesibilitas Geografis:** Jarak ke Denpasar adalah {distance:.2f} km (Jauh/Outer Region). "
            "Lokasi ini membutuhkan waktu tempuh yang lama, memerlukan sarana transportasi yang memadai untuk dikunjungi."
        )
        
    return investor_insight, gov_insight, spatial_desc

def update_coords():
    """Callback untuk mengubah titik Lat/Lon ketika Kabupaten diganti dari Dropdown"""
    kab = st.session_state.kab_selector
    for key, (lat, lon) in KABUPATEN_COORDS.items():
        if key.lower() in kab.lower():
            st.session_state.input_lat = lat
            st.session_state.input_lon = lon
            break
def load_example_coords(df_ref):
    """Callback untuk mengisi koordinat contoh dan data terkait ke session state"""
    random_row = df_ref.sample(n=1).iloc[0]
    st.session_state.input_lat = float(random_row['latitude'])
    st.session_state.input_lon = float(random_row['longitude'])
    st.session_state.selected_kat = random_row['kategori']
    st.session_state.kab_selector = random_row['kabupaten_kota']
    st.session_state.example_name = random_row['nama']
    st.session_state.example_lat = float(random_row['latitude'])
    st.session_state.example_lon = float(random_row['longitude'])

# --- UI Layout ---
def main():
    load_custom_css()
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>Bali Tourism Decision Support System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 40px;'>Prediksi Tingkat Kepopuleran Destinasi Wisata dengan Support Vector Machine (SVM)</p>", unsafe_allow_html=True)
    
    with st.spinner("Memuat model dan matriks spasial..."):
        svm_model, le_kategori, le_kabupaten, scaler, categories, cities, df_ref = load_dependencies()
        
    # Inisialisasi State Awal
    if 'input_lat' not in st.session_state:
        st.session_state.input_lat = -8.6500
    if 'input_lon' not in st.session_state:
        st.session_state.input_lon = 115.2167
    if 'selected_kat' not in st.session_state:
        st.session_state.selected_kat = categories[0]
    if 'kab_selector' not in st.session_state:
        st.session_state.kab_selector = cities[0]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🗺️ Data Geografis & Kategori")
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        selected_kategori = st.selectbox("Pilih Kategori Wisata", categories, key='selected_kat')
        
        # Dropdown Kabupaten memicu fungsi callback update_coords saat diubah
        selected_kabupaten = st.selectbox("Pilih Kabupaten/Kota", cities, key='kab_selector', on_change=update_coords)
        
        c_lat, c_lon = st.columns(2)
        with c_lat:
            # Value dikendalikan oleh session state
            lat = st.number_input("Latitude", key='input_lat', format="%.5f")
        with c_lon:
            lon = st.number_input("Longitude", key='input_lon', format="%.5f")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_btn = st.button("Analisis Kepopuleran 🚀")
        with col_btn2:
            st.button("🎲 Contoh Objek Wisata", on_click=load_example_coords, args=(df_ref,))
            
        if 'example_name' in st.session_state:
            if (st.session_state.input_lat != st.session_state.get('example_lat') or 
                st.session_state.input_lon != st.session_state.get('example_lon')):
                del st.session_state.example_name
            else:
                st.info(f"📍 Contoh: **{st.session_state.example_name}**")
        
    with col2:
        st.markdown("### 📍 Lokasi Objek Wisata (Peta Bali)")
        
        # Temukan spot terdekat berdasarkan lat/lon saat ini
        nearest_spot = find_nearest_spot(lat, lon, selected_kabupaten, df_ref)
        ref_lat = nearest_spot['latitude']
        ref_lon = nearest_spot['longitude']
        
        # Peta ter-center di titik Lat/Lon saat ini
        m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
        
        bali_bounds = [[-8.9, 114.4], [-8.0, 115.7]]
        m.fit_bounds(bali_bounds)
        m.options['minZoom'] = 9
        m.options['maxBounds'] = bali_bounds
        
        # Marker 1: Input manual pengguna (Merah)
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>Lokasi Input Anda</b><br>Lokasi: {lat:.5f}, {lon:.5f}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        # Marker 2: Objek wisata nyata terdekat (Biru)
        folium.Marker(
            location=[ref_lat, ref_lon],
            popup=f"<b>Referensi Terdekat:</b> {nearest_spot['nama']}<br>Kategori: {nearest_spot['kategori']}<br>Lokasi: {ref_lat:.5f}, {ref_lon:.5f}",
            icon=folium.Icon(color="blue", icon="star")
        ).add_to(m)
        
        # Garis penghubung jika jaraknya cukup signifikan
        if nearest_spot['distance_km'] > 0.05:
            folium.PolyLine(
                locations=[[lat, lon], [ref_lat, ref_lon]],
                color="#3b82f6",
                weight=2.5,
                dash_array='5, 5',
                tooltip=f"Jarak ke Referensi: {nearest_spot['distance_km']:.2f} km"
            ).add_to(m)
        
        st_folium(m, width=700, height=350, returned_objects=[])
        
    st.markdown("---")
    
    if predict_btn:
        st.markdown("<h3 style='text-align: center;'>📊 Hasil Prediksi (Pendekatan Spasial Terdekat)</h3>", unsafe_allow_html=True)
        
        with st.spinner("Melakukan inferensi spasial..."):
            # Memproyeksikan koordinat input ke koordinat nyata objek terdekat
            nearest_spot = find_nearest_spot(lat, lon, selected_kabupaten, df_ref)
            ref_lat = nearest_spot['latitude']
            ref_lon = nearest_spot['longitude']
            
            # Hitung jarak dari Denpasar berdasarkan koordinat nyata terdekat
            dist_ref = haversine_distance(DENPASAR_LAT, DENPASAR_LON, ref_lat, ref_lon)
            kat_encoded = le_kategori.transform([selected_kategori])[0]
            kab_encoded = le_kabupaten.transform([selected_kabupaten])[0]
            
            x_df = pd.DataFrame(
                [[kat_encoded, kab_encoded, ref_lat, ref_lon, dist_ref]], 
                columns=['kategori_encoded', 'kabupaten_encoded', 'latitude', 'longitude', 'DistanceToCenter']
            )
            x_scaled = scaler.transform(x_df)
            
            prediction = svm_model.predict(x_scaled)[0]
            probabilities = svm_model.predict_proba(x_scaled)[0]
            
            if prediction == 2:
                kelas_teks = "SEMPURNA (Rating 5.0)"
                css_class = "result-sempurna"
            elif prediction == 1:
                kelas_teks = "BAGUS (Rating 4.5 - 4.9)"
                css_class = "result-bagus"
            else:
                kelas_teks = "KURANG (Rating < 4.5)"
                css_class = "result-kurang"
                
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                st.markdown(f"""
                <div class='{css_class}' style='height: auto; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 25px;'>
                    <div>{kelas_teks}</div>
                    <div style='font-size: 14px; font-weight: normal; margin-top: 10px; opacity: 0.8;'>
                        Referensi Terproyeksi: {nearest_spot['nama']} ({nearest_spot['distance_km']:.2f} km)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tambahkan penjelasan kualitatif pariwisata (DSS Insights) di bawah banner hasil
                investor_insight, gov_insight, spatial_desc = generate_explanation(
                    prediction, dist_ref, selected_kategori, selected_kabupaten
                )
                
                st.markdown("<div class='glass-container' style='margin-top: 20px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top:0;'>💡 Analisis Keputusan & Wawasan Spasial (DSS Insights)</h4>", unsafe_allow_html=True)
                st.markdown(investor_insight, unsafe_allow_html=True)
                st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                st.markdown(gov_insight, unsafe_allow_html=True)
                st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                st.markdown(spatial_desc, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_res2:
                st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
                st.markdown("#### Detail Probabilitas")
                st.write(f"🟢 Sempurna: **{probabilities[2]*100:.1f}%**")
                st.write(f"🔵 Bagus: **{probabilities[1]*100:.1f}%**")
                st.write(f"🔴 Kurang: **{probabilities[0]*100:.1f}%**")
                st.markdown("---")
                st.write(f"📍 **Objek Referensi:** {nearest_spot['nama']}")
                st.write(f"📏 **Jarak ke Referensi:** {nearest_spot['distance_km']:.2f} km")
                st.write(f"🏢 **Jarak ke Denpasar (Pusat):** {dist_ref:.2f} km")
                st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # SECTION: TABEL PERBANDINGAN BATCH PREDICTION PER KABUPATEN
    # ================================================================
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>🗃️ Tabel Perbandingan Semua Objek Wisata per Kabupaten</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Model SVM memproyeksikan seluruh objek wisata di kabupaten terpilih sekaligus berdasarkan koordinat dan kategorinya.</p>", unsafe_allow_html=True)

    col_filter1, col_filter2 = st.columns([1, 1])
    with col_filter1:
        batch_kabupaten = st.selectbox("📌 Pilih Kabupaten untuk Perbandingan", cities, key='batch_kab')
    with col_filter2:
        label_filter_options = ["Semua Kelas", "🟢 Sempurna", "🔵 Bagus", "🔴 Kurang"]
        batch_label_filter = st.selectbox("🔍 Filter Hasil Prediksi", label_filter_options, key='batch_label_filter')

    batch_btn = st.button("🔄 Jalankan Batch Prediction", key='batch_btn')

    if batch_btn:
        with st.spinner(f"Memproses seluruh data wisata di {batch_kabupaten}..."):
            # Ambil semua data dari kabupaten yang dipilih
            df_batch = df_ref[df_ref['kabupaten_kota'].str.strip().str.lower() == batch_kabupaten.strip().lower()].copy()
            df_batch = df_batch.dropna(subset=['latitude', 'longitude'])

            if df_batch.empty:
                st.warning("Tidak ada data wisata untuk kabupaten yang dipilih.")
            else:
                results = []
                for _, row in df_batch.iterrows():
                    try:
                        dist = haversine_distance(DENPASAR_LAT, DENPASAR_LON, row['latitude'], row['longitude'])
                        kat_enc = le_kategori.transform([row['kategori']])[0]
                        kab_enc = le_kabupaten.transform([row['kabupaten_kota']])[0]

                        x = pd.DataFrame(
                            [[kat_enc, kab_enc, row['latitude'], row['longitude'], dist]],
                            columns=['kategori_encoded', 'kabupaten_encoded', 'latitude', 'longitude', 'DistanceToCenter']
                        )
                        x_sc = scaler.transform(x)
                        pred = svm_model.predict(x_sc)[0]
                        prob = svm_model.predict_proba(x_sc)[0]

                        if pred == 2:
                            label = "🟢 Sempurna"
                        elif pred == 1:
                            label = "🔵 Bagus"
                        else:
                            label = "🔴 Kurang"

                        results.append({
                            "Nama Objek Wisata": row['nama'],
                            "Kategori": row['kategori'],
                            "Latitude": round(row['latitude'], 5),
                            "Longitude": round(row['longitude'], 5),
                            "Jarak ke Denpasar (km)": round(dist, 2),
                            "Hasil Prediksi SVM": label,
                            "Prob. Sempurna (%)": round(prob[2] * 100, 1),
                            "Prob. Bagus (%)": round(prob[1] * 100, 1),
                            "Prob. Kurang (%)": round(prob[0] * 100, 1),
                        })
                    except Exception:
                        continue

                df_result = pd.DataFrame(results)

                # Terapkan filter label jika dipilih
                if batch_label_filter != "Semua Kelas":
                    df_result = df_result[df_result["Hasil Prediksi SVM"] == batch_label_filter]

                # Ringkasan Statistik Kabupaten
                df_all_results = pd.DataFrame(results)
                sempurna_count = (df_all_results["Hasil Prediksi SVM"] == "🟢 Sempurna").sum()
                bagus_count    = (df_all_results["Hasil Prediksi SVM"] == "🔵 Bagus").sum()
                kurang_count   = (df_all_results["Hasil Prediksi SVM"] == "🔴 Kurang").sum()
                total_count    = len(df_all_results)

                st.markdown(f"""
                <div style='display: flex; gap: 16px; margin-bottom: 20px;'>
                    <div style='flex:1; padding: 16px; border-radius: 12px; background: rgba(16,185,129,0.12); border: 1px solid #10b981; text-align: center;'>
                        <div style='font-size: 28px; font-weight: bold; color: #34d399;'>{sempurna_count}</div>
                        <div style='color: #94a3b8; font-size: 13px;'>🟢 Sempurna</div>
                    </div>
                    <div style='flex:1; padding: 16px; border-radius: 12px; background: rgba(59,130,246,0.12); border: 1px solid #3b82f6; text-align: center;'>
                        <div style='font-size: 28px; font-weight: bold; color: #60a5fa;'>{bagus_count}</div>
                        <div style='color: #94a3b8; font-size: 13px;'>🔵 Bagus</div>
                    </div>
                    <div style='flex:1; padding: 16px; border-radius: 12px; background: rgba(239,68,68,0.12); border: 1px solid #ef4444; text-align: center;'>
                        <div style='font-size: 28px; font-weight: bold; color: #f87171;'>{kurang_count}</div>
                        <div style='color: #94a3b8; font-size: 13px;'>🔴 Kurang</div>
                    </div>
                    <div style='flex:1; padding: 16px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); text-align: center;'>
                        <div style='font-size: 28px; font-weight: bold; color: #e2e8f0;'>{total_count}</div>
                        <div style='color: #94a3b8; font-size: 13px;'>📊 Total Data</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if df_result.empty:
                    st.info(f"Tidak ada objek wisata dengan prediksi **{batch_label_filter}** di {batch_kabupaten}.")
                else:
                    st.dataframe(
                        df_result,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Hasil Prediksi SVM": st.column_config.TextColumn("Prediksi SVM", width="medium"),
                            "Prob. Sempurna (%)": st.column_config.ProgressColumn("Sempurna %", min_value=0, max_value=100, format="%.1f%%"),
                            "Prob. Bagus (%)": st.column_config.ProgressColumn("Bagus %", min_value=0, max_value=100, format="%.1f%%"),
                            "Prob. Kurang (%)": st.column_config.ProgressColumn("Kurang %", min_value=0, max_value=100, format="%.1f%%"),
                        }
                    )

                    # Tombol unduh CSV
                    csv_export = df_result.copy()
                    csv_export["Hasil Prediksi SVM"] = csv_export["Hasil Prediksi SVM"].str.replace(r'[🟢🔵🔴]\s*', '', regex=True)
                    csv_bytes = csv_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Unduh Hasil sebagai CSV",
                        data=csv_bytes,
                        file_name=f"prediksi_wisata_{batch_kabupaten.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key='download_csv'
                    )

if __name__ == "__main__":
    main()
