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
        
        /* Example Tourism Spot Card style */
        .example-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-left: 5px solid #8b5cf6;
        }
        .example-card h5 {
            margin: 0 0 5px 0 !important;
            color: #60a5fa !important;
            font-size: 16px !important;
            background: none !important;
            -webkit-text-fill-color: initial !important;
        }
        .example-card p {
            margin: 3px 0 !important;
            font-size: 13px !important;
            color: #cbd5e1 !important;
        }
        .rating-badge {
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
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

# Jumlah objek wisata terdekat yang ditampilkan sebagai referensi
TOP_N_SPOTS = 5

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
    
    df_ref = df[['nama', 'kategori', 'kabupaten_kota', 'latitude', 'longitude', 'rating']].copy()
    df_ref['latitude'] = pd.to_numeric(df_ref['latitude'], errors='coerce')
    df_ref['longitude'] = pd.to_numeric(df_ref['longitude'], errors='coerce')
    df_ref['rating'] = pd.to_numeric(df_ref['rating'], errors='coerce')
    df_ref = df_ref.dropna()
    
    # Filter sensor kata-kata konyol / troll di Google Maps agar aman saat sidang TA
    troll_pattern = r'berak|kencing|troll|sampah|berak masal|toilet'
    df_ref = df_ref[~df_ref['nama'].str.lower().str.contains(troll_pattern, na=False)]
    
    df_clean = df.drop(columns=['nama', 'preferensi', 'link'])
    df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
    df_clean = df_clean.dropna()
    
    df_clean['DistanceToCenter'] = df_clean.apply(
        lambda row: haversine_distance(DENPASAR_LAT, DENPASAR_LON, row['latitude'], row['longitude']),
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

def find_top_n_spots(target_lat, target_lon, df_ref, n=TOP_N_SPOTS):
    """Mencari top-N objek wisata nyata terdekat secara global berdasarkan jarak Haversine"""
    df_temp = df_ref.copy()
    df_temp['_dist_km'] = df_temp.apply(
        lambda row: haversine_distance(target_lat, target_lon, row['latitude'], row['longitude']),
        axis=1
    )

    top_n = df_temp.nsmallest(n, '_dist_km').reset_index(drop=True)

    results = []
    for _, row in top_n.iterrows():
        results.append({
            'nama': row['nama'],
            'kategori': row['kategori'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'rating': row['rating'],
            'kabupaten_kota': row['kabupaten_kota'],
            'distance_km': row['_dist_km']
        })
    return results

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
    # Reset spot selector and example card when kabupaten changes
    st.session_state.spot_selector = "-- Pilih Contoh Objek Wisata --"
    for key in ['example_name', 'example_rating', 'example_lat', 'example_lon', 'example_kategori', 'example_kabupaten']:
        if key in st.session_state:
            del st.session_state[key]

def load_selected_spot_coords():
    """Callback untuk mengisi koordinat dari objek wisata terpilih dari dropdown"""
    selected_spot_name = st.session_state.spot_selector
    if selected_spot_name != "-- Pilih Contoh Objek Wisata --" and 'df_ref_global' in st.session_state:
        df_ref = st.session_state.df_ref_global
        spot_rows = df_ref[df_ref['nama'] == selected_spot_name]
        if not spot_rows.empty:
            spot_row = spot_rows.iloc[0]
            st.session_state.input_lat = float(spot_row['latitude'])
            st.session_state.input_lon = float(spot_row['longitude'])
            st.session_state.selected_kat = spot_row['kategori']
            st.session_state.example_name = spot_row['nama']
            st.session_state.example_rating = spot_row['rating']
            st.session_state.example_lat = float(spot_row['latitude'])
            st.session_state.example_lon = float(spot_row['longitude'])
            st.session_state.example_kategori = spot_row['kategori']
            st.session_state.example_kabupaten = spot_row['kabupaten_kota']
    else:
        for key in ['example_name', 'example_rating', 'example_lat', 'example_lon', 'example_kategori', 'example_kabupaten']:
            if key in st.session_state:
                del st.session_state[key]

def load_example_coords(df_ref):
    """Callback untuk mengisi koordinat contoh dan data terkait ke session state"""
    random_row = df_ref.sample(n=1).iloc[0]
    st.session_state.kab_selector = random_row['kabupaten_kota']
    st.session_state.input_lat = float(random_row['latitude'])
    st.session_state.input_lon = float(random_row['longitude'])
    st.session_state.selected_kat = random_row['kategori']
    st.session_state.example_name = random_row['nama']
    st.session_state.example_rating = random_row['rating']
    st.session_state.example_lat = float(random_row['latitude'])
    st.session_state.example_lon = float(random_row['longitude'])
    st.session_state.example_kategori = random_row['kategori']
    st.session_state.example_kabupaten = random_row['kabupaten_kota']
    
    st.session_state.spot_selector = random_row['nama']

# --- UI Layout ---
def main():
    load_custom_css()
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>Bali Tourism Decision Support System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 40px;'>Prediksi Tingkat Kepopuleran Destinasi Wisata dengan Support Vector Machine (SVM)</p>", unsafe_allow_html=True)
    
    with st.spinner("Memuat model dan matriks spasial..."):
        svm_model, le_kategori, le_kabupaten, scaler, categories, cities, df_ref = load_dependencies()
    
    st.session_state.df_ref_global = df_ref
        
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
        
        # Dropdown Objek Wisata Referensi di Kabupaten Terpilih
        df_spots = df_ref[df_ref['kabupaten_kota'] == selected_kabupaten]
        spot_names = ["-- Pilih Contoh Objek Wisata --"] + sorted(df_spots['nama'].unique().tolist())
            
        selected_spot = st.selectbox(
            "🔎 Pilih Contoh dari Objek Wisata Nyata", 
            spot_names, 
            key='spot_selector', 
            on_change=load_selected_spot_coords
        )
        
        c_lat, c_lon = st.columns(2)
        with c_lat:
            # Value dikendalikan oleh session state, presisi ditingkatkan (step=1e-7)
            lat = st.number_input("Latitude", key='input_lat', format="%.7f", step=0.0000001)
        with c_lon:
            lon = st.number_input("Longitude", key='input_lon', format="%.7f", step=0.0000001)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_btn = st.button("Analisis Kepopuleran 🚀")
        with col_btn2:
            st.button("🎲 Contoh Objek Wisata", on_click=load_example_coords, args=(df_ref,))
            
        if 'example_name' in st.session_state:
            if (st.session_state.input_lat != st.session_state.get('example_lat') or 
                st.session_state.input_lon != st.session_state.get('example_lon')):
                # Reset if user manually modifies coordinates
                for key in ['example_name', 'example_rating', 'example_lat', 'example_lon', 'example_kategori', 'example_kabupaten']:
                    if key in st.session_state:
                        del st.session_state[key]
            else:
                rating_val = st.session_state.get('example_rating', 'N/A')
                if isinstance(rating_val, (int, float)) and not pd.isna(rating_val):
                    rating_str = f"{rating_val:.1f}"
                else:
                    rating_str = str(rating_val)
                st.markdown(f"""
                <div class='example-card'>
                    <h5>📍 Objek Wisata Referensi: {st.session_state.example_name}</h5>
                    <p><b>Kategori:</b> {st.session_state.get('example_kategori', '-')} | <b>Wilayah:</b> {st.session_state.get('example_kabupaten', '-')}</p>
                    <p><b>Rating Aktual:</b> <span class='rating-badge'>⭐ {rating_str}</span> (Google Maps)</p>
                    <p style='font-size: 11px; opacity: 0.6; margin-top: 5px !important;'>Koordinat: {st.session_state.example_lat:.5f}, {st.session_state.example_lon:.5f}</p>
                </div>
                """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### 📍 Lokasi Objek Wisata (Peta Bali)")

        # Temukan top-N spot terdekat secara global berdasarkan lat/lon saat ini
        top_spots = find_top_n_spots(lat, lon, df_ref)

        # Peta ter-center di titik Lat/Lon saat ini
        m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")

        bali_bounds = [[-8.9, 114.4], [-8.0, 115.7]]
        m.options['minZoom'] = 9
        m.options['maxBounds'] = bali_bounds

        # Marker: Input manual pengguna (Merah)
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>Lokasi Input Anda</b><br>Koordinat: {lat:.7f}, {lon:.7f}",
            tooltip="📌 Lokasi Input Anda",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

        # Warna marker untuk top-N (peringkat 1 = biru terang, dst.)
        spot_colors = ["blue", "darkblue", "cadetblue", "lightblue", "purple"]

        for rank, spot in enumerate(top_spots):
            color = spot_colors[rank] if rank < len(spot_colors) else "gray"
            r_val = spot['rating']
            r_str = f"{r_val:.1f}" if isinstance(r_val, (int, float)) and not pd.isna(r_val) else str(r_val)
            label = " ★ Referensi Utama" if rank == 0 else ""

            folium.Marker(
                location=[spot['latitude'], spot['longitude']],
                popup=(
                    f"<b>#{rank+1}{label}</b><br>"
                    f"<b>{spot['nama']}</b><br>"
                    f"Wilayah: {spot['kabupaten_kota']}<br>"
                    f"Kategori: {spot['kategori']}<br>"
                    f"Rating: ⭐ {r_str}<br>"
                    f"Jarak dari input: {spot['distance_km']:.2f} km"
                ),
                tooltip=f"#{rank+1} {spot['nama']} ({spot['distance_km']:.2f} km)",
                icon=folium.Icon(color=color, icon="star")
            ).add_to(m)

            # Garis putus-putus dari input ke masing-masing spot
            if spot['distance_km'] > 0.05:
                # Gambar garis hubung
                folium.PolyLine(
                    locations=[[lat, lon], [spot['latitude'], spot['longitude']]],
                    color="#2563eb" if rank == 0 else "#64748b",
                    weight=3 if rank == 0 else 1.5,
                    dash_array='5, 5',
                    tooltip=f"Jarak ke #{rank+1} {spot['nama']}: {spot['distance_km']:.2f} km"
                ).add_to(m)

                # Tambahkan label teks jarak di titik tengah garis
                mid_lat = (lat + spot['latitude']) / 2
                mid_lon = (lon + spot['longitude']) / 2
                
                # Format label dengan latar belakang putih semi-transparan agar mudah dibaca
                label_color = "#2563eb" if rank == 0 else "#334155"
                border_color = "#3b82f6" if rank == 0 else "#94a3b8"
                bg_opacity = 0.9 if rank == 0 else 0.8
                
                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=folium.DivIcon(
                        html=f"""
                            <div style="
                                font-family: 'Inter', sans-serif;
                                font-size: 10px;
                                color: {label_color};
                                background-color: rgba(255, 255, 255, {bg_opacity});
                                border: 1px solid {border_color};
                                border-radius: 4px;
                                padding: 2px 5px;
                                white-space: nowrap;
                                font-weight: bold;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.15);
                                transform: translate(-50%, -50%);
                            ">
                                {spot['distance_km']:.2f} km
                            </div>
                        """
                    )
                ).add_to(m)

        # Fit bounds agar fokus pada input dan referensi terdekat
        points = [[lat, lon]] + [[spot['latitude'], spot['longitude']] for spot in top_spots]
        m.fit_bounds(points)

        st_folium(m, width=700, height=350, returned_objects=[])
        
    st.markdown("---")
    
    if predict_btn:
        st.markdown("<h3 style='text-align: center;'>📊 Hasil Prediksi (Pendekatan Spasial Terdekat)</h3>", unsafe_allow_html=True)
        
        with st.spinner("Melakukan inferensi spasial..."):
            # Ambil top-N spot terdekat secara global
            top_spots = find_top_n_spots(lat, lon, df_ref)
            # Gunakan spot #1 (terdekat) sebagai referensi utama
            nearest_spot = top_spots[0]
            detected_kabupaten = nearest_spot['kabupaten_kota']

            # Hitung jarak dari Denpasar berdasarkan koordinat input pengguna aktual
            dist_input = haversine_distance(DENPASAR_LAT, DENPASAR_LON, lat, lon)
            kat_encoded = le_kategori.transform([selected_kategori])[0]
            kab_encoded = le_kabupaten.transform([detected_kabupaten])[0]

            # Gunakan koordinat input user secara langsung untuk prediksi
            x_df = pd.DataFrame(
                [[kat_encoded, kab_encoded, lat, lon, dist_input]],
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
                # Tampilkan info jika wilayah terdeteksi otomatis berbeda dengan selectbox dropdown
                warn_html = ""
                if detected_kabupaten.strip().lower() != selected_kabupaten.strip().lower():
                    warn_html = f"""
                    <div style='
                        background: rgba(245, 158, 11, 0.15);
                        border: 1px solid #f59e0b;
                        color: #fbbf24;
                        padding: 10px 15px;
                        border-radius: 8px;
                        margin-bottom: 15px;
                        font-size: 13px;
                        text-align: left;
                    '>
                        ⚠️ <b>Deteksi Lokasi:</b> Koordinat berada di wilayah <b>{detected_kabupaten}</b>. 
                        Sistem menggunakan wilayah ini agar input model SVM sesuai dengan koordinat aktual.
                    </div>
                    """

                st.markdown(f"""
                {warn_html}
                <div class='{css_class}' style='height: auto; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 25px;'>
                    <div>{kelas_teks}</div>
                    <div style='font-size: 14px; font-weight: normal; margin-top: 10px; opacity: 0.8;'>
                        Referensi Utama: {nearest_spot['nama']} ({nearest_spot['distance_km']:.2f} km dari input)
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Tambahkan penjelasan kualitatif pariwisata (DSS Insights) di bawah banner hasil
                investor_insight, gov_insight, spatial_desc = generate_explanation(
                    prediction, dist_input, selected_kategori, detected_kabupaten
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
                # Panel Probabilitas
                st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
                st.markdown("#### Detail Probabilitas")
                st.write(f"🟢 Sempurna: **{probabilities[2]*100:.1f}%**")
                st.write(f"🔵 Bagus: **{probabilities[1]*100:.1f}%**")
                st.write(f"🔴 Kurang: **{probabilities[0]*100:.1f}%**")
                st.markdown("---")
                st.write(f"🏢 **Jarak ke Denpasar:** {dist_input:.2f} km")
                st.markdown("</div>", unsafe_allow_html=True)

                # Panel Top-N Referensi Terproyeksi
                st.markdown("<div class='glass-container' style='margin-top: 15px;'>", unsafe_allow_html=True)
                st.markdown(f"#### 📍 Top {TOP_N_SPOTS} Referensi Terdekat")
                st.caption(f"Objek wisata nyata paling dekat dari titik input Anda.")
                rank_colors = ["🔵", "🔹", "🩵", "🔷", "💠"]
                for rank, spot in enumerate(top_spots):
                    r_val = spot['rating']
                    r_str = f"{r_val:.1f}" if isinstance(r_val, (int, float)) and not pd.isna(r_val) else str(r_val)
                    icon = rank_colors[rank] if rank < len(rank_colors) else "▪️"
                    label = " ★ Utama" if rank == 0 else ""
                    border_color = "#3b82f6" if rank == 0 else "#475569"
                    bg_opacity = 0.12 if rank == 0 else 0.05
                    st.markdown(
                        f"""
                        <div style='
                            background: rgba(255,255,255,{bg_opacity});
                            border-radius: 8px;
                            padding: 10px 12px;
                            margin-bottom: 8px;
                            border-left: 3px solid {border_color};
                        '>
                            <div style='font-size:13px; font-weight:bold; color:#e2e8f0;'>
                                {icon} #{rank+1} {spot['nama']}
                                <span style='font-size:10px; color:#94a3b8; font-weight:normal;'>{label}</span>
                            </div>
                            <div style='font-size:10px; color:#cbd5e1; margin-top:2px; font-style:italic;'>
                                Wilayah: {spot['kabupaten_kota']}
                            </div>
                            <div style='font-size:11px; color:#94a3b8; margin-top:3px;'>
                                📂 {spot['kategori']} &nbsp;|&nbsp;
                                ⭐ {r_str} &nbsp;|&nbsp;
                                📏 {spot['distance_km']:.2f} km
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
