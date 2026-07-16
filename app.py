import json
import os
import re
import time
import base64
import html
import io
import asyncio
import logging
from urllib.parse import quote_plus, urlparse, parse_qs
from urllib.request import urlopen
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from textblob import TextBlob
from database import get_connection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from tornado.websocket import WebSocketClosedError, WebSocketProtocol13

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

def silence_closed_websocket_tasks():
    if getattr(WebSocketProtocol13.write_message, "_agrisentiment_patched", False):
        return

    original_write_message = WebSocketProtocol13.write_message

    def write_message_without_unretrieved_warning(self, *args, **kwargs):
        future = original_write_message(self, *args, **kwargs)

        def consume_expected_close_error(done_future):
            try:
                exception = done_future.exception()
            except WebSocketClosedError:
                pass
            except asyncio.CancelledError:
                pass
            else:
                if exception is not None:
                    logging.getLogger(__name__).warning(
                        "Unexpected websocket write error",
                        exc_info=(type(exception), exception, exception.__traceback__),
                    )

        if hasattr(future, "add_done_callback"):
            future.add_done_callback(consume_expected_close_error)
        return future

    write_message_without_unretrieved_warning._agrisentiment_patched = True
    WebSocketProtocol13.write_message = write_message_without_unretrieved_warning

silence_closed_websocket_tasks()

load_dotenv()

# =============================
# POSTGRESQL DATA CONFIG
# =============================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.getenv("DB_NAME", "agrisentiment")

DATA_TABLES = {
    "best_model_per_video": "data/Hasil_Best_Model_Setiap_Video.csv",
    "sentiment_per_video": "data/Sentiment_per_Video.csv",
    "hasil_sentimen": "data/Hasil_Sentimen.csv",
    "sentiment_per_topik": "data/Sentiment_per_Topik.csv",
    "mapping": "data/Mapping.csv",
    "result_svm_balancing": "result_svm_balancing.csv",
    "result_nb_balancing": "result_NB_balancing.csv",
    "result_lstm_balancing": "result_lstm_balancing.csv",
    "kamuskatabaku": "kamuskatabaku.xlsx",
}

ASSET_FILES = [
    "Perkebunan-Rakyat.jpeg",
    "dashboard.png",
    "emotional.png",
    "classification.png",
    "smile.png",
    "neutral.png",
    "sad.png",
]

def local_path(relative_path):
    return os.path.join(APP_DIR, relative_path)

def get_postgres_engine():
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")

    if not db_password:
        raise Exception("DB_PASSWORD belum diatur di file .env")

    return create_engine(
        f"postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{DB_NAME}"
    )

def ensure_postgres_data():
    engine = get_postgres_engine()
    inspector = inspect(engine)

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS app_assets (
                name TEXT PRIMARY KEY,
                content_type TEXT,
                data BYTEA NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("ALTER TABLE app_assets ADD COLUMN IF NOT EXISTS data BYTEA"))
        conn.execute(text("ALTER TABLE app_assets ADD COLUMN IF NOT EXISTS content_type TEXT"))
        conn.execute(text("ALTER TABLE app_assets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
    existing_tables = set(inspector.get_table_names())
    for table_name, source_file in DATA_TABLES.items():
        if table_name in existing_tables:
            continue

        source_path = local_path(source_file)
        if os.path.exists(source_path):
            if source_path.lower().endswith((".xlsx", ".xls")):
                df_source = pd.read_excel(source_path)
            else:
                df_source = pd.read_csv(source_path, encoding="utf-8")
            df_source.to_sql(table_name, engine, if_exists="replace", index=False)

    image_assets = set(ASSET_FILES)
    for file_name in os.listdir(APP_DIR):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_assets.add(file_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            for asset_name in image_assets:
                cur.execute("SELECT data FROM app_assets WHERE name=%s", (asset_name,))
                existing_asset = cur.fetchone()
                if existing_asset and existing_asset[0] is not None:
                    continue

                source_path = local_path(asset_name)
                if os.path.exists(source_path):
                    content_type = "image/jpeg" if asset_name.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    with open(source_path, "rb") as asset_file:
                        asset_data = psycopg2.Binary(asset_file.read())
                        cur.execute(
                            """
                            UPDATE app_assets
                            SET content_type=%s,
                                data=%s,
                                updated_at=CURRENT_TIMESTAMP
                            WHERE name=%s
                            """,
                            (content_type, asset_data, asset_name)
                        )

                        if cur.rowcount == 0:
                            cur.execute(
                                """
                                INSERT INTO app_assets (name, content_type, data)
                                VALUES (%s, %s, %s)
                                """,
                                (asset_name, content_type, asset_data)
                            )
            conn.commit()

@st.cache_data(show_spinner=False)
def load_table_from_postgres(table_name):
    engine = get_postgres_engine()
    return pd.read_sql_table(table_name, engine)

@st.cache_data(show_spinner=False)
def load_asset_from_postgres(asset_name):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM app_assets WHERE name=%s", (asset_name,))
            row = cur.fetchone()

    if not row:
        raise FileNotFoundError(f"Asset {asset_name} belum tersedia di PostgreSQL.")

    return bytes(row[0])

ensure_postgres_data()

# =============================
# LOAD IMAGE FROM POSTGRESQL
# =============================
def get_base64_image(image_file):
    data = load_asset_from_postgres(image_file)
    return base64.b64encode(data).decode()

bg_image = get_base64_image("Perkebunan-Rakyat.jpeg")
img_bibit = get_base64_image("bibit.png")

def get_base64_icon(path):
    return get_base64_image(path)


def get_optional_base64_asset(path):
    try:
        return get_base64_image(path)
    except FileNotFoundError:
        return None

icon_dashboard = get_base64_icon("dashboard.png")
icon_sentiment = get_base64_icon("emotional.png")
icon_classification = get_base64_icon("classification.png")
icon_smile = get_base64_icon("smile.png")
icon_neutral = get_base64_icon("neutral.png")   
icon_sad = get_base64_icon("sad.png")
icon_realtime = get_base64_icon("realtime.png")

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="AgriSentiment",
    layout="wide"
)

st.markdown(f"""
<style>

/* FULL BACKGROUND DASHBOARD */
.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* DARK OVERLAY (FIX) */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.85);
    z-index: 0;
}}

/* pastikan semua konten di atas overlay */
.block-container, .dashboard-header, section[data-testid="stSidebar"] {{
    position: relative;
    z-index: 1;
}}

</style>
""", unsafe_allow_html=True)

# =============================
# PUBLIC LANDING PAGE
# =============================
if st.query_params.get("landing") == "1":
    st.session_state.app_started = False
    st.session_state.start_param_handled = False

if st.query_params.get("start") == "1":
    first_start = not st.session_state.get("app_started", False)
    st.session_state.app_started = True
    if first_start or not st.session_state.get("start_param_handled", False):
        st.session_state.menu = "Dashboard"
        st.session_state.start_param_handled = True
    st.query_params.clear()
    st.rerun()

if not st.session_state.get("app_started", False):

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }}
    header, footer {{ visibility: hidden; }}
    .stApp {{
        background: #04130c !important;
        font-family: 'Inter', sans-serif;
    }}
    .stApp::before {{ display: none !important; 
            background: rgba(0,0,0,0.20);}}
    .landing-shell {{
        min-height: 100vh;
        color: white;
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
    }}
    .landing-header {{
        height: 55px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 max(22px, calc((100vw - 1260px) / 2));
        background: rgba(3, 27, 12, 0.55);
        border-bottom: 1px solid rgba(186,230,195,0.15);
        backdrop-filter: blur(16px);
        position: fixed;
        inset: 0 0 auto 0;
        z-index: 20;
    }}
    .landing-brand {{
        font-size: 25px;
        font-weight: 900;
        letter-spacing: 0;
        color: white !important;
        text-decoration: none !important;
    }}
    .landing-brand span {{ color: #34d399; }}
    .landing-nav {{
        display: flex;
        align-items: center;
        gap: 20px;
        font-size: 13px;
        font-weight: 800;
    }}
    .landing-nav a {{
        color: rgba(236,253,245,0.88) !important;
        text-decoration: none !important;
    }}
    .landing-nav a:hover,
    .landing-nav a.active,
    .landing-shell:has(#fitur:target) .landing-nav a[href="#fitur"],
    .landing-shell:has(#model-evaluasi:target) .landing-nav a[href="#model-evaluasi"],
    .landing-shell:has(#about-dashboard:target) .landing-nav a[href="#about-dashboard"] {{
        color: #34d399 !important;
    }}
    .landing-hero {{
        min-height: 100vh;
        padding: 92px max(22px, calc((100vw - 1260px) / 2)) 64px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(360px, 0.95fr);
        gap: 54px;
        align-items: center;
        background:
            radial-gradient(circle at 76% 28%, rgba(20,184,166,0.18), transparent 32%),
            linear-gradient(115deg, #030906 0%, #06170f 50%, #03110b 100%);
        color: white;
        scroll-margin-top: 60px;
    }}
    .hero-badge {{
        display: inline-flex !important;
        width: max-content !important;
        max-width: 100% !important;
        background: rgba(5, 123, 87, 0.52) !important;
        border: 1px solid rgba(52,211,153,0.34) !important;
        color: #6ee7b7 !important;
        font-size: 12px !important;
        letter-spacing: 0.02em !important;
        padding: 8px 14px !important;
        margin-top: -50px !important;
        margin-bottom: 30px !important;
        border-radius: 999px !important;
    }}
    .hero-title {{
        font-size: clamp(44px, 5vw, 62px);
        line-height: 1.08;
        font-weight: 950;
        letter-spacing: 0;
        margin-top: -10px;
        margin-bottom: -10px;
        max-width: 720px;
        color: #ffffff;
    }}
    .hero-title span {{
        display: block;
        color: #34d399;
    }}
    .hero-desc {{
        max-width: 590px;
        color: rgba(236,253,245,0.82);
        font-size: 15px;
        line-height: 1.65;
        font-weight: 800;
        margin-top: -70px;
        margin-bottom: 20px;
    }}
    .hero-visual {{
        min-height: 400px;
        border-radius: 8px;
        background:
            linear-gradient(90deg, rgba(3,9,6,0.58), rgba(3,17,11,0.18) 44%, rgba(2,6,5,0.32)),
            radial-gradient(circle at 55% 34%, rgba(34,197,94,0.36), transparent 34%),
            url("data:image/jpeg;base64,{img_bibit}") center/cover no-repeat;
        box-shadow: 0 30px 80px rgba(0,0,0,0.28);
        margin-top: -30px;
    }}
    .start-button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 48px;
        padding: 0 26px;
        border-radius: 6px;
        border: 1px solid rgba(134,239,172,0.35);
        background: #0aa36f;
        color: white !important;
        text-decoration: none !important;
        font-weight: 900;
        box-shadow: 0 14px 32px rgba(34,197,94,0.26);
    }}
    .start-button:hover {{
        background: #22c55e;
        color: white !important;
    }}
    .landing-section {{
        min-height: 100vh;
        padding: 92px max(22px, calc((100vw - 1260px) / 2)) 72px;
        background: rgba(0,0,0,0.34);
        border-top: 1px solid rgba(255,255,255,0.08);
        border-bottom: 1px solid rgba(255,255,255,0.06);
        scroll-margin-top: 50px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .landing-section > *,
    .mission-section > * {{
        transform: translateY(-34px);
    }}
    .section-heading {{
        text-align: center;
        margin-bottom: 34px;
    }}
    .section-heading h2 {{
        color: #10b981;
        font-size: 28px;
        margin: -20px 0 8px;
        font-weight: 950;
    }}
    .section-heading p {{
        color: rgba(220,252,231,0.70);
        font-size: 13px;
        font-weight: 800;
        margin: 0;
    }}
    .feature-grid, .tech-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 24px;
    }}
    .feature-grid {{
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 14px 20px;
    }}
    .feature-card, .tech-card {{
        background: rgba(255,255,255,0.055);
        border-radius: 14px;
        padding: 28px;
        box-shadow: 0 16px 44px rgba(0,0,0,0.20);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 176px;
    }}
    .feature-card {{
        grid-column: span 2;
        min-height: 138px;
        padding: 18px 20px;
    }}
    .feature-card:nth-child(4) {{
        grid-column: 2 / span 2;
    }}
    .feature-card:nth-child(5) {{
        grid-column: 4 / span 2;
    }}
    .feature-card .feature-icon {{
        width: 32px;
        height: 32px;
        font-size: 16px;
        margin-bottom: 10px;
    }}
    .feature-card h4 {{
        margin-bottom: 7px;
        font-size: 15px;
    }}
    .feature-card p {{
        font-size: 11px;
        line-height: 1.45;
    }}
    .feature-icon, .tech-icon {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(5, 123, 87, 0.70);
        color: #ecfdf5;
        font-size: 18px;
        font-weight: 950;
        margin-bottom: 16px;
    }}
    .feature-card h4, .tech-card h4 {{
        color: white !important;
        margin: 0 0 10px;
        font-size: 16px;
        font-weight: 950;
    }}
    .feature-card p, .tech-card p {{
        color: rgba(220,252,231,0.70) !important;
        font-size: 12px;
        line-height: 1.7;
        margin: 0;
    }}
    #fitur .section-heading {{
        margin-bottom: 22px;
    }}
    #fitur .feature-icon {{
        width: 32px;
        height: 32px;
        font-size: 16px;
        margin-bottom: 8px;
    }}
    #fitur .feature-card h4 {{
        margin: 0 0 7px;
        font-size: 15px;
        margin-bottom: 2px;
    }}
    #fitur .feature-card p {{
        font-size: 11px;
        line-height: 1.45;
    }}
    .mission-section {{
        display: grid;
        grid-template-columns: minmax(340px, 0.9fr) minmax(0, 1fr);
        gap: 52px;
        align-items: center;
        min-height: 100vh;
        padding: 92px max(22px, calc((100vw - 1260px) / 2)) 72px;
        background: rgba(6, 32, 17, 0.88);
        border-top: 1px solid rgba(255,255,255,0.08);
        scroll-margin-top: 50px;
    }}
    .mission-image {{
        min-height: 360px;
        border-radius: 8px;
        background: url("data:image/jpeg;base64,{bg_image}") center/cover no-repeat;
        box-shadow: 0 18px 44px rgba(15,23,42,0.14);
    }}
    .mission-copy h2 {{
        color: #dcfce7;
        font-size: 28px;
        line-height: 1.2;
        margin: 0 0 18px;
        font-weight: 950;
    }}
    .mission-copy p {{
        color: rgba(220,252,231,0.78);
        font-size: 14px;
        line-height: 1.8;
        margin-top: -10px;
        text-align: justify;
        text-justify: inter-word;
    }}
    .mission-points {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
        margin-top: 22px;
    }}
    .mission-point {{
        border-left: 3px solid #65a30d;
        padding-left: 12px;
        color: rgba(220,252,231,0.72);
        font-size: 12px;
        line-height: 1.55;
    }}
    .mission-point b {{
        display: block;
        color: #86efac;
        margin-bottom: 4px;
    }}
    .landing-footer {{
        text-align: center;
        background: rgba(3, 27, 12, 0.55);
        color: rgba(220,252,231,0.62);
        font-size: 12px;
        padding: 18px;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-bottom: -50px;
    }}
    @media (max-width: 800px) {{
        .landing-header {{ padding: 0 18px; }}
        .landing-nav {{ gap: 12px; font-size: 11px; }}
        .landing-hero, .feature-grid, .tech-grid, .mission-section {{ grid-template-columns: 1fr; }}
        .feature-card,
        .feature-card:nth-child(4),
        .feature-card:nth-child(5) {{ grid-column: auto; }}
        .landing-hero, .landing-section, .mission-section {{ padding: 96px 22px 56px; }}
        .landing-section > *,
        .mission-section > * {{ transform: translateY(-18px); }}
        .hero-visual, .mission-image {{ min-height: 200px; }}
        .mission-points {{ grid-template-columns: 1fr; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-shell">
        <div class="landing-header">
            <a class="landing-brand" href="#landing"><span>🌾 Agri</span>Sentiment</a>
            <div class="landing-nav">
                <a href="#fitur">Fitur</a>
                <a href="#model-evaluasi">Model Evaluasi</a>
                <a href="#about">Tentang</a>
            </div>
        </div>
        <section class="landing-hero" id="landing">
            <div>
                <div class="hero-badge">ANALISIS OPINI PUBLIK BERBASIS NLP</div>
                <h1 class="hero-title">Pahami Sentimen <span>Pertanian Modern</span></h1>
                <p class="hero-desc">
                    AgriSentiment membantu memetakan persepsi publik dari komentar YouTube
                    bertema pertanian melalui preprocessing teks, klasifikasi sentimen,
                    evaluasi model, dan visualisasi insight yang siap dianalisis.
                </p>
                <a class="start-button" href="?start=1" target="_self">Mulai Sekarang</a>
            </div>
            <div class="hero-visual" aria-label="Visual analitik dashboard AgriSentiment"></div>
        </section>
    """, unsafe_allow_html=True)

    st.markdown("""
        <section class="landing-section" id="fitur">
            <div class="section-heading">
                <h2>Fitur Unggulan Kami</h2>
                <p>AgriSentiment membantu membaca komentar publik tentang pertanian agar lebih mudah dipahami, dibandingkan, dan dijadikan bahan evaluasi.</p>
            </div>
            <div class="feature-grid">
                <div class="feature-card">
                    <span class="feature-icon">📊</span>
                    <h4>Dashboard Interaktif</h4>
                    <p>Menyediakan visualisasi komprehensif mengenai tren komentar, distribusi sentimen, serta analisis sentimen berdasarkan topik pertanian untuk mendukung eksplorasi dan interpretasi data secara interaktif.</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">🔎</span>
                    <h4>Analisis Sentimen</h4>
                    <p>Menyajikan analisis sentimen secara mendalam pada setiap video berdasarkan topik dan kategori pertanian yang dipilih, dilengkapi distribusi sentimen, statistik komentar, serta ringkasan insight untuk memahami persepsi publik secara lebih detail.</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">🏷️</span>
                    <h4>Realtime Analysis</h4>
                    <p>Menganalisis komentar YouTube secara langsung menggunakan URL atau ID video sehingga pengguna dapat memperoleh hasil sentimen dan statistik komentar secara instan.</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">⚖️</span>
                    <h4>Evaluasi Model</h4>
                    <p>Membandingkan performa Naive Bayes, Support Vector Machine (SVM), dan Long Short-Term Memory (LSTM) menggunakan metrik akurasi, presisi, recall, dan F1-score.</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">☁️</span>
                    <h4>Analisis Kata Kunci</h4>
                    <p>Mengidentifikasi kata-kata yang paling sering muncul pada setiap video atau topik pertanian melalui word cloud dan frekuensi kata, sehingga pengguna dapat memahami fokus pembahasan serta karakteristik sentimen positif, netral, dan negatif.</p>
                </div>
            </div>
        </section>
        <section class="landing-section" id="model-evaluasi">
            <div class="section-heading">
                <h2>Model Evaluasi</h2>
                <p>Setiap model diuji untuk melihat seberapa baik program mengenali pola bahasa pada komentar pertanian dan menentukan kelas sentimennya.</p>
            </div>
            <div class="tech-grid">
                <div class="tech-card">
                    <span class="tech-icon">🤖</span>
                    <h4>Support Vector Machine</h4>
                    <p>Membandingkan performa klasifikasi sentimen menggunakan pendekatan hyperplane optimal untuk memisahkan komentar ke dalam kategori sentimen yang berbeda. Model ini dievaluasi untuk melihat kemampuannya dalam menghasilkan klasifikasi yang akurat dan konsisten.</p>
                </div>
                <div class="tech-card">
                    <span class="tech-icon">⚙️</span>
                    <h4>Naive Bayes</h4>
                    <p>Membandingkan performa klasifikasi sentimen menggunakan pendekatan probabilistik berbasis frekuensi kemunculan kata pada komentar. Model ini dikenal efisien dalam memproses data teks dan digunakan sebagai salah satu benchmark dalam evaluasi hasil klasifikasi.</p>
                </div>
                <div class="tech-card">
                    <span class="tech-icon">🧠</span>
                    <h4>Long Short-Term Memory</h4>
                    <p>Membandingkan performa klasifikasi sentimen dengan memanfaatkan kemampuan jaringan saraf dalam memahami konteks dan hubungan antar kata pada komentar. Model ini digunakan untuk menangkap pola sentimen yang lebih kompleks dibandingkan metode klasifikasi lainnya.</p>
                </div>
            </div>
        </section>
        <section class="mission-section" id="about">
            <div class="mission-image"></div>
            <div class="mission-copy">
                <h2>🌾 AgriSentiment</h2>
                <p>AgriSentiment merupakan platform analisis sentimen berbasis Machine Learning yang dirancang khusus untuk pegiat pertanian dapat mengeksplorasi komentar video YouTube pada berbagai topik seperti pertanian budidaya, pupuk, irigasi, hidroponik, produk organik, dan pengendalian hama.</p>
                <p>Dengan menggabungkan analisis sentimen, pemetaan topik, eksplorasi kata kunci, dan evaluasi model klasifikasi dalam satu platform, AgriSentiment memberikan wawasan yang lebih komprehensif untuk memahami opini publik secara cepat dan mudah dipahami.</p>
                <div class="mission-points">
                    <div class="mission-point"><b>Keunggulan</b>Mengintegrasikan analisis sentimen, topik, dan kata kunci dalam satu platform sehingga pengguna dapat memahami opini publik dari berbagai sudut pandang secara lebih komprehensif.</div>
                    <div class="mission-point"><b>Manfaat</b>Membantu pengguna menemukan informasi penting dari ribuan komentar tanpa perlu membaca seluruh komentar secara manual.</div>
                </div>
            </div>
        </section>
        <div class="landing-footer">
            &copy; 2026 AgriSentiment | Developed by Puput Ayu Setiawati - Politeknik Elektronika Negeri Surabaya
        </div>
    </div>
    """, unsafe_allow_html=True)

    components.html("""
    <script>
    const doc = window.parent.document;
    const navLinks = Array.from(doc.querySelectorAll('.landing-nav a'));
    const sections = navLinks
        .map((link) => doc.querySelector(link.getAttribute('href')))
        .filter(Boolean);

    function setActiveNav() {
        if (!navLinks.length || !sections.length) return;

        const current = sections.reduce((active, section) => {
            const rect = section.getBoundingClientRect();
            return rect.top <= 180 ? section : active;
        }, null);

        navLinks.forEach((link) => {
            const isActive = current && link.getAttribute('href') === `#${current.id}`;
            link.classList.toggle('active', isActive);
            link.style.setProperty('color', isActive ? '#34d399' : 'rgba(236,253,245,0.88)', 'important');
        });
    }

    navLinks.forEach((link) => {
        link.addEventListener('click', () => {
            navLinks.forEach((item) => {
                item.classList.remove('active');
                item.style.setProperty('color', 'rgba(236,253,245,0.88)', 'important');
            });
            link.classList.add('active');
            link.style.setProperty('color', '#34d399', 'important');
            window.setTimeout(setActiveNav, 350);
        });
    });

    doc.addEventListener('scroll', setActiveNav, { passive: true });
    window.parent.addEventListener('scroll', setActiveNav, { passive: true });
    window.parent.addEventListener('hashchange', setActiveNav);
    setActiveNav();
    </script>
    """, height=0)

    st.stop()

col1, col2, col3 = st.columns([1, 20, 1])

with col1:
    if st.button("☰", key="toggle_sidebar"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

with col3:
    pass
        
# =============================
# HEADER DASHBOARD
# =============================
st.markdown("""
<style>

/* sembunyikan header default */
header[data-testid="stHeader"]{
    display:none;
}

/* HEADER DASHBOARD */
.dashboard-header{
    position:fixed;
    top:0;
    left:0;
    width:100%;
    height:50px;
    background: #053616;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 30px;
    z-index:1000;
    box-shadow:0 2px 10px rgba(0,0,0,0.15);
}

.header-left{
    top:0;
    font-size:25px;
    font-weight:bold;
    color:white !important;
    display:flex;
    align-items:center;
    gap:8px;
    margin-left:30px;
    text-decoration:none !important;
}

/* tombol hamburger */
.menu-btn button{
    background:none;
    border:none;
    color:white;
    font-size:22px;
    cursor:pointer;
    padding:0;
}

.menu-btn button:hover{
    opacity:0.8;
}

/* logo */
.header-title{
    color:white;
    font-size:24px;
    font-weight:700;
}

/* user kanan */
.header-user{
    color:white;
    font-size:15px;
    display:flex;
    align-items:center;
    gap:8px;
}

.header-logout{
    color:#f8fffb !important;
    background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.22);
    border-radius:8px;
    padding:7px 12px;
    font-size:13px;
    font-weight:700;
    line-height:1;
    text-decoration:none !important;
    display:inline-flex;
    align-items:center;
    gap:6px;
    transition:all 0.2s ease;
    margin-right: -13px;
    margin-left: 8px;
}

.header-logout:hover{
    background:#ffffff;
    color:#053616 !important;
    border-color:#ffffff;
    transform:translateY(-1px);
    box-shadow:0 6px 14px rgba(0,0,0,0.18);
}

/* jarak konten dari header */
.block-container{
    padding-top:90px;
}

/* SIDEBAR UTAMA */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #053616 0%, #064b22 100%);
    width: 240px !important;
    min-width: 240px !important;
    padding-top: 20px;
}

section[data-testid="stSidebar"] > div {
    padding-bottom: 24px;
}

/* TEXT SIDEBAR */
section[data-testid="stSidebar"] * {
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
}

section[data-testid="stSidebar"] h3 {
    margin-top: 0 !important;
    margin-bottom: 14px !important;
    padding-bottom: 0 !important;
    transform: translateY(-8px);
}

/* MENU BUTTON - Kondisi Normal */
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    height: 40px !important;
    min-height: 40px !important;
    background: transparent !important;
    border: 1px solid rgba(220,252,231,0.28) !important;
    text-align: left !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 0 14px !important;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease !important;
}

section[data-testid="stSidebar"] .stButton {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    color: #ecfdf5 !important;
    font-size: 14px !important;
    font-weight: 650 !important;
    line-height: 1 !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] div:has(.sidebar-menu-anchor) + div .stButton > button::before,
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.sidebar-menu-anchor) + div .stButton > button::before {
    content: "" !important;
    width: 18px !important;
    height: 18px !important;
    flex: 0 0 18px !important;
    display: inline-block !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #0b7a3b !important;
    border-color: rgba(220,252,231,0.42) !important;
    transform: translateY(-1px);
}

/* DESKRIPSI */
.menu-desc {
    font-size: 12px;
    margin-top: 2px;
    margin-bottom: 8px;
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    color: #dcfce7;
    border-left: 3px solid #86efac;
}

.sidebar-menu-anchor {
    display: none;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] div:has(> .sidebar-menu-anchor) {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] div:has(> .sidebar-menu-anchor) + div .stButton,
section[data-testid="stSidebar"] div:has(.sidebar-menu-anchor) + div .stButton,
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.sidebar-menu-anchor) + div .stButton {
    margin-bottom: -4px !important;
}

/* REVISI UTAMA: Background tombol berubah hijau saat menu terpilih/aktif */
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] div:has(> .sidebar-menu-anchor.active) + div .stButton > button,
section[data-testid="stSidebar"] div:has(.sidebar-menu-anchor.active) + div .stButton > button,
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.sidebar-menu-anchor.active) + div .stButton > button {
    background: #0b7a3b !important;
    border-color: rgba(220,252,231,0.42) !important;
}

section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #0b7a3b !important;
    border-color: rgba(220,252,231,0.42) !important;
}

div[data-testid="column"]:first-child button {
    position: fixed;
    top: 5px;
    left: 15px;
    z-index: 1100;
    background: rgba(255,255,255,0.21);
    border: none;
    color: white;
    font-size: 20px;
    font-weight: bold;
    width: 38px;
    height: 38px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: 0.2s ease;
}

/* HOVER EFFECT */
div[data-testid="column"]:first-child button:hover {
    background: rgba(255,255,255,0.3);
    transform: scale(1.05);
}

/* ACTIVE (saat diklik) */
div[data-testid="column"]:first-child button:active {
    transform: scale(0.95);
}

div[data-testid="column"]:nth-child(3) button {
    position: fixed !important;
    top: 7px !important;
    right: 24px !important;
    z-index: 1300 !important;
    width: 34px !important;
    min-width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    border-radius: 50% !important;
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
}

div[data-testid="column"]:nth-child(3) button p {
    display: none !important;
}

div[data-testid="column"]:nth-child(3) button::before {
    content: "" !important;
    width: 26px !important;
    height: 26px !important;
    display: block !important;
    border-radius: 50% !important;
    border: 2px solid rgba(255,255,255,0.35) !important;
    background-image: url("https://cdn2.iconfinder.com/data/icons/user-people-4/48/5-512.png") !important;
    background-size: cover !important;
    background-position: center !important;
    background-color: white !important;
}

div[data-testid="column"]:nth-child(3) button:hover {
    background: transparent !important;
    transform: scale(1.05) !important;
}

div[data-testid="column"]:nth-child(3) button:hover::before {
    border-color: #86efac !important;
}
                        
</style>
""", unsafe_allow_html=True)

if st.query_params.get("landing") == "1":
    st.query_params.clear()
    st.session_state.app_started = False
    st.session_state.start_param_handled = False
    st.rerun()

# =============================
# SIDEBAR TOGGLE STATE
# =============================
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

st.markdown(f"""
<div class="dashboard-header">

<a class="header-left" href="?landing=1" target="_self">
🌾 <div class="header-title">AgriSentiment</div>
</a>

<div class="header-user">
<b>Hi, Selamat Datang</b>
<a class="header-logout" href="?landing=1" target="_self" aria-label="Keluar ke landing page">Keluar</a>
</div>

</div>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
if st.session_state.sidebar_open:

    st.sidebar.markdown("### Menu")

    st.sidebar.markdown(f"""
    <style>
    section[data-testid="stSidebar"] div:has(.sidebar-menu-anchor.menu-dashboard) + div .stButton > button::before,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.sidebar-menu-anchor.menu-dashboard) + div .stButton > button::before {{
        background-image: url("data:image/png;base64,{icon_dashboard}") !important;
    }}
    section[data-testid="stSidebar"] div:has(.sidebar-menu-anchor.menu-sentiment) + div .stButton > button::before,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.sidebar-menu-anchor.menu-sentiment) + div .stButton > button::before {{
        background-image: url("data:image/png;base64,{icon_sentiment}") !important;
    }}
    section[data-testid="stSidebar"] div:has(.sidebar-menu-anchor.menu-realtime) + div .stButton > button::before,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.sidebar-menu-anchor.menu-realtime) + div .stButton > button::before {{
        background-image: url("data:image/png;base64,{icon_realtime}") !important;
    }}
    section[data-testid="stSidebar"] div:has(.sidebar-menu-anchor.menu-model) + div .stButton > button::before,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.sidebar-menu-anchor.menu-model) + div .stButton > button::before {{
        background-image: url("data:image/png;base64,{icon_classification}") !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    if "menu" not in st.session_state:
        st.session_state.menu = "Dashboard"

    def sidebar_menu_item(title, desc, key_name, icon_base64):
        is_active = st.session_state.menu == key_name
        active_class = " active" if is_active else ""
        menu_icon_class = {
            "Dashboard": "menu-dashboard",
            "Sentiment Analysis": "menu-sentiment",
            "Realtime Analysis": "menu-realtime",
            "Model Evaluation": "menu-model",
        }.get(key_name, "")

        st.sidebar.markdown(
            f'<div class="sidebar-menu-anchor {menu_icon_class}{active_class}"></div>',
            unsafe_allow_html=True
        )

        if st.sidebar.button(
            title,
            key=key_name,
            type="primary" if is_active else "secondary",
            use_container_width=True
        ):
            st.session_state.menu = key_name
            st.rerun()

        if is_active:
            st.sidebar.markdown(f"""
            <div class="menu-desc">
                {desc}
            </div>
            """, unsafe_allow_html=True)


    sidebar_menu_item(
        "Dashboard",
        "Ringkasan interaktif data sentimen secara keseluruhan, meliputi distribusi sentimen, termasuk metadata video, serta tren periode waktu yang dapat dipantau.",
        "Dashboard",
        icon_dashboard
    )

    sidebar_menu_item(
        "Sentiment Analysis",
        "Eksplorasi mendalam hasil analisis sentimen per video. Memungkinkan identifikasi distribusi sentimen spesifik untuk mendukung pengambilan keputusan riset.",
        "Sentiment Analysis",
        icon_sentiment
    )

    sidebar_menu_item(
        "Realtime Analysis",
        "Analisis instan komentar YouTube melalui link atau ID video. Memberikan hasil klasifikasi sentimen serta statistik frekuensi kata secara langsung.",
        "Realtime Analysis",
        icon_realtime
    )

    sidebar_menu_item(
        "Model Evaluation",
        "Perbandingan performa model klasifikasi (SVM, Naive Bayes, LSTM) berdasarkan metrik akurasi, presisi, recall, dan F1-Score.",
        "Model Evaluation",
        icon_classification 
    )

    menu = st.session_state.menu

else:
    menu = st.session_state.get("menu", "Dashboard")

# =========================================================
# ======================= DASHBOARD =======================
# =========================================================

# LOAD DATA
@st.cache_data
def load_model_data():
    return load_table_from_postgres("best_model_per_video")

@st.cache_data
def load_sentiment_video():
    return load_table_from_postgres("sentiment_per_video")

@st.cache_data
def load_sentiment():
    return load_table_from_postgres("hasil_sentimen")

@st.cache_data
def load_sentiment_topik():
    return load_table_from_postgres("sentiment_per_topik")

df_model = load_model_data()
df_sentiment_video = load_sentiment_video()
df_sentiment_topik = load_sentiment_topik()
df_sentiment = load_sentiment()
 
# NORMALISASI VIDEO_ID
df_model['video_id'] = df_model['video_id'].astype(str).str.strip()
df_sentiment_video['video_id'] = df_sentiment_video['video_id'].astype(str).str.strip()
df_sentiment['video_id'] = df_sentiment['video_id'].astype(str).str.strip()
df_sentiment_topik['topic']=df_sentiment_topik['topic'].astype(str).str.strip()


SENTIMENT_LABELS = ["Positif", "Netral", "Negatif"]
SENTIMENT_EXAMPLE_KEYWORDS = {
    "Positif": {
        "bagus", "baik", "mantap", "hebat", "keren", "bermanfaat", "suka", "setuju",
        "menarik", "mudah", "efektif", "membantu", "sukses", "terbaik", "recommended"
    },
    "Netral": {
        "info", "informasi", "cara", "berapa", "kapan", "dimana", "bagaimana", "apa",
        "mohon", "tanya", "video", "pak", "bu", "min", "tolong"
    },
    "Negatif": {
        "mahal", "susah", "sulit", "gagal", "buruk", "jelek", "rugi", "kurang",
        "masalah", "bahaya", "ribet", "tidak", "nggak", "ga", "gak", "jangan"
    }
}


COMMON_KEYWORD_STOPWORDS = {
    "yg", "ya", "yang", "dan", "di", "ke", "dari", "itu", "ini", "ada", "adalah",
    "nya", "nih", "dong", "sih", "aja", "juga", "untuk", "dengan", "pada", "atau",
    "karena", "jadi", "kalau", "kalo", "pak", "bu", "min", "video"
}
SENTIMENT_KEYWORD_ALIASES = {
    "positif": "positive",
    "positive": "positive",
    "netral": "neutral",
    "neutral": "neutral",
    "negatif": "negative",
    "negative": "negative",
}


@st.cache_data(show_spinner=False)
def load_sentiment_keyword_lexicon():
    try:
        df_kw = pd.read_csv(local_path("sentiment_keywords.csv"))
    except Exception:
        return pd.DataFrame(columns=["topic", "sentiment", "keyword"])

    required_cols = {"topic", "sentiment", "keyword"}
    if not required_cols.issubset(df_kw.columns):
        return pd.DataFrame(columns=["topic", "sentiment", "keyword"])

    df_kw = df_kw[list(required_cols)].copy()
    for col in required_cols:
        df_kw[col] = df_kw[col].fillna("").astype(str).str.lower().str.strip()
    df_kw["sentiment"] = df_kw["sentiment"].map(lambda value: SENTIMENT_KEYWORD_ALIASES.get(value, value))
    df_kw = df_kw[
        (df_kw["sentiment"].isin({"positive", "neutral", "negative"}))
        & (df_kw["keyword"] != "")
        & (df_kw["keyword"] != "nan")
    ].drop_duplicates()
    return df_kw


def normalize_sentiment_label(label):
    return SENTIMENT_KEYWORD_ALIASES.get(str(label).lower().strip(), str(label).lower().strip())


def normalize_topic_filter(topic_filter):
    if topic_filter is None:
        return None
    if isinstance(topic_filter, str):
        topics = [topic_filter]
    else:
        topics = list(topic_filter)
    topics = [str(topic).lower().strip() for topic in topics if str(topic).strip()]
    if not topics or "semua topik" in topics:
        return None
    return set(topics)


def get_sentiment_top_keywords(df, sentiment_label, n=3, topic_filter=None):
    if df is None or df.empty or "sentiment" not in df.columns:
        return ["-"]

    target_sentiment = normalize_sentiment_label(sentiment_label)
    work_df = df.copy()
    sentiment_series = work_df["sentiment"].fillna("").astype(str).map(normalize_sentiment_label)
    work_df = work_df[sentiment_series == target_sentiment].copy()

    topics = normalize_topic_filter(topic_filter)
    if topics and "topic" in work_df.columns:
        topic_series = work_df["topic"].fillna("").astype(str).str.lower().str.strip()
        work_df = work_df[topic_series.isin(topics)]

    if work_df.empty:
        return ["-"]

    text_cols = [
        col for col in ["stemming", "stopword", "normalization", "comment", "raw_comment", "Komentar", "text"]
        if col in work_df.columns
    ]
    if not text_cols:
        return ["-"]

    combined_text = " ".join(
        work_df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .tolist()
    )
    combined_text = re.sub(r"\s+", " ", combined_text).strip()
    if not combined_text:
        return ["-"]

    lexicon = load_sentiment_keyword_lexicon()
    lexicon = lexicon[lexicon["sentiment"] == target_sentiment].copy()
    if topics and not lexicon.empty:
        lexicon = lexicon[lexicon["topic"].isin(topics)]

    keyword_counts = []
    for keyword in lexicon["keyword"].dropna().astype(str).str.lower().str.strip().drop_duplicates():
        if not keyword or keyword in COMMON_KEYWORD_STOPWORDS:
            continue
        pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
        count = len(re.findall(pattern, combined_text))
        if count > 0:
            keyword_counts.append((keyword, count, len(keyword.split())))

    keyword_counts = sorted(keyword_counts, key=lambda item: (item[1], item[2], item[0]), reverse=True)
    selected = []
    for keyword, _, _ in keyword_counts:
        if keyword not in selected:
            selected.append(keyword)
        if len(selected) >= n:
            return selected

    single_word_lexicon = {
        keyword for keyword in lexicon["keyword"].dropna().astype(str).str.lower().str.strip()
        if keyword and " " not in keyword and keyword not in COMMON_KEYWORD_STOPWORDS
    }
    single_word_lexicon.update(SENTIMENT_EXAMPLE_KEYWORDS.get(str(sentiment_label).title(), set()))
    single_word_lexicon = {word.lower().strip() for word in single_word_lexicon if word}

    try:
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        X = vectorizer.fit_transform([combined_text])
        df_freq = pd.DataFrame({
            "Kata": vectorizer.get_feature_names_out(),
            "Frekuensi": X.toarray().flatten()
        })
        df_freq = df_freq[
            (~df_freq["Kata"].isin(COMMON_KEYWORD_STOPWORDS))
            & (df_freq["Kata"].isin(single_word_lexicon))
        ].sort_values(by="Frekuensi", ascending=False)

        for word in df_freq["Kata"].tolist():
            if word not in selected:
                selected.append(word)
            if len(selected) >= n:
                break
    except Exception:
        pass

    return selected if selected else ["-"]


def get_sentiment_comment_examples(df, sentiment_label, limit=3):
    if df is None or df.empty or "sentiment" not in df.columns:
        return []

    comment_col = next((col for col in ["raw_comment", "comment", "Komentar", "text"] if col in df.columns), None)
    if not comment_col:
        return []

    author_col = next((col for col in ["author", "Author", "username", "name", "Nama"] if col in df.columns), None)
    sentiment_series = df["sentiment"].fillna("").astype(str).str.lower().str.strip()
    mask = sentiment_series == str(sentiment_label).lower().strip()
    example_df = df.loc[mask, [comment_col] + ([author_col] if author_col else [])].copy()
    example_df[comment_col] = (
        example_df[comment_col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    example_df = example_df[
        (example_df[comment_col] != "")
        & (example_df[comment_col].str.lower() != "nan")
    ].drop_duplicates(subset=[comment_col])
    example_df["_length_score"] = example_df[comment_col].str.len()
    keywords = SENTIMENT_EXAMPLE_KEYWORDS.get(sentiment_label, set())
    example_df["_keyword_score"] = example_df[comment_col].str.lower().apply(
        lambda comment: sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", comment))
    )
    example_df = example_df.sort_values(
        by=["_keyword_score", "_length_score"],
        ascending=[False, False]
    ).head(limit)

    examples = []
    for _, row in example_df.iterrows():
        author = "Tidak diketahui"
        if author_col:
            author = str(row.get(author_col, "")).strip() or "Tidak diketahui"
            if author.lower() == "nan":
                author = "Tidak diketahui"
        examples.append({
            "author": author,
            "comment": row[comment_col]
        })
    return examples


def render_sentiment_comment_dropdown(df, key_prefix, title="Contoh Kalimat Komentar", empty_message=None):
    st.markdown(f"""
    <div class="visual-container sentiment-example-header">
        <div class="visual-title">{title}</div>
        <div class="visual-subtitle">Pilih sentimen untuk melihat contoh komentar yang mewakili hasil analisis</div>
    </div>
    <div class="sentiment-example-select-anchor"></div>
    """, unsafe_allow_html=True)

    selected_sentiment = st.selectbox(
        "Pilih Sentimen",
        options=SENTIMENT_LABELS,
        index=0,
        key=f"{key_prefix}_sentiment_comment",
        label_visibility="collapsed"
    )
    examples = get_sentiment_comment_examples(df, selected_sentiment)

    if examples:
        for idx, example in enumerate(examples, start=1):
            st.markdown(f"""
            <div class="comment-example-card">
                <div class="comment-example-meta">
                    <span class="comment-example-label">{selected_sentiment} #{idx}</span>
                    <span class="comment-example-author">Author: {html.escape(example["author"])}</span>
                </div>
                <div class="comment-example-text">{html.escape(example["comment"])}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        message = empty_message or f"Belum ada contoh komentar untuk sentimen {html.escape(selected_sentiment)} pada data yang sedang ditampilkan."
        st.markdown(f"""
        <div class="comment-example-empty">
            {message}
        </div>
        """, unsafe_allow_html=True)


def extract_youtube_video_id(value):
    value = str(value or "").strip()
    if not value:
        return ""

    if re.fullmatch(r"[\w-]{11}", value):
        return value

    parsed = urlparse(value)
    if parsed.netloc:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            return query_id.strip()

        path_parts = [part for part in parsed.path.split("/") if part]
        if "youtu.be" in parsed.netloc and path_parts:
            return path_parts[0].strip()
        if "shorts" in path_parts:
            shorts_index = path_parts.index("shorts")
            if len(path_parts) > shorts_index + 1:
                return path_parts[shorts_index + 1].strip()

    match = re.search(r"([\w-]{11})", value)
    return match.group(1) if match else ""


def youtube_api_get(endpoint, params):
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY belum diatur di file .env.")

    params = {**params, "key": api_key}
    query = "&".join(f"{key}={quote_plus(str(val))}" for key, val in params.items() if val is not None)
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?{query}"
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_youtube_video(video_id):
    data = youtube_api_get("videos", {"part": "snippet", "id": video_id})
    items = data.get("items", [])
    if not items:
        raise ValueError("Video tidak ditemukan atau tidak dapat diakses.")
    snippet = items[0].get("snippet", {})
    return {
        "title": snippet.get("title") or f"Video {video_id}",
        "channel": snippet.get("channelTitle") or "Unknown Channel",
        "published_at": snippet.get("publishedAt"),
    }


def fetch_youtube_comment_replies(parent_id):
    replies = []
    page_token = None

    while True:
        data = youtube_api_get(
            "comments",
            {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": 100,
                "textFormat": "plainText",
                "pageToken": page_token,
            }
        )

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            replies.append({
                "author": snippet.get("authorDisplayName", "Unknown"),
                "comment": snippet.get("textOriginal", ""),
                "like_count": int(snippet.get("likeCount", 0) or 0),
                "date": snippet.get("publishedAt", ""),
            })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return replies


def fetch_youtube_comments(video_id):
    comments = []
    page_token = None

    while True:
        data = youtube_api_get(
            "commentThreads",
            {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "textFormat": "plainText",
                "pageToken": page_token,
                "order": "time",
            }
        )

        for item in data.get("items", []):
            thread_snippet = item.get("snippet", {})
            top_level_comment = thread_snippet.get("topLevelComment", {})
            top_level_comment_id = top_level_comment.get("id")
            snippet = top_level_comment.get("snippet", {})
            comments.append({
                "author": snippet.get("authorDisplayName", "Unknown"),
                "comment": snippet.get("textOriginal", ""),
                "like_count": int(snippet.get("likeCount", 0) or 0),
                "date": snippet.get("publishedAt", ""),
            })

            if top_level_comment_id and int(thread_snippet.get("totalReplyCount", 0) or 0) > 0:
                comments.extend(fetch_youtube_comment_replies(top_level_comment_id))

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments


@st.cache_data(show_spinner=False)
def load_realtime_keyword_data():
    kamus_data = {}
    kamus_tables = ("kamus_kata", "kamuskata", "kamus_kata_baku", "kamuskatabaku")
    for table_name in kamus_tables:
        try:
            df_kamus = load_table_from_postgres(table_name)
            if {"tidak_baku", "kata_baku"}.issubset(df_kamus.columns):
                kamus_data = dict(zip(
                    df_kamus["tidak_baku"].astype(str).str.lower().str.strip(),
                    df_kamus["kata_baku"].astype(str).str.lower().str.strip()
                ))
                kamus_data = {key: value for key, value in kamus_data.items() if key and key != "nan"}
                break
        except Exception:
            continue

    kamus_path = local_path("data/kamuskata.csv")
    if not os.path.exists(kamus_path):
        kamus_path = local_path("kamuskatabaku.xlsx")

    try:
        if not kamus_data:
            if kamus_path.lower().endswith(".csv"):
                df_kamus = pd.read_csv(kamus_path)
            else:
                df_kamus = pd.read_excel(kamus_path)
            if {"tidak_baku", "kata_baku"}.issubset(df_kamus.columns):
                kamus_data = dict(zip(
                    df_kamus["tidak_baku"].astype(str).str.lower().str.strip(),
                    df_kamus["kata_baku"].astype(str).str.lower().str.strip()
                ))
                kamus_data = {key: value for key, value in kamus_data.items() if key and key != "nan"}
    except Exception:
        if not kamus_data:
            kamus_data = {}

    topic_keywords = pd.read_csv(local_path("topic_keywords.csv"))
    category_keywords = pd.read_csv(local_path("kategori_keywords.csv"))
    sentiment_keywords = pd.read_csv(local_path("sentiment_keywords.csv"))
    return kamus_data, topic_keywords, category_keywords, sentiment_keywords


def clean_realtime_text(text):
    text = "" if pd.isna(text) else str(text)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"['\n&\\#]", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def case_fold_realtime_text(text):
    return "" if pd.isna(text) else str(text).lower()


def tokenize_realtime_text(text):
    return ", ".join(str(text).split())


def normalize_realtime_tokens(text, kamus_data):
    tokens = [word.strip().lower() for word in str(text).split(", ") if word.strip()]
    return ", ".join(kamus_data.get(word, word) for word in tokens)


def simple_stem_text(text):
    stemmer = get_realtime_stemmer()
    text = re.sub(r"\bng([a-z]+)", r"\1", text)
    text = re.sub(r"\bke([a-z]+)", r"\1", text)
    text = re.sub(r"\bdi([a-z]+)", r"\1", text)
    text = re.sub(r"\bmeng([a-z]+)", r"\1", text)
    text = re.sub(r"nya\b", "", text)
    words = [word.strip() for word in text.split(",")]
    if stemmer:
        words = [stemmer.stem(word) for word in words]
    return ", ".join(words)


@st.cache_resource(show_spinner=False)
def get_realtime_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_realtime_stopwords():
    fallback_stopwords = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "ada", "untuk", "dengan",
        "saya", "kami", "kita", "nya", "ya", "yg", "the", "a", "an"
    }
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("indonesian"))
    except Exception:
        return fallback_stopwords


def identify_realtime_topic(text, topic_keywords):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()
    scores = {}
    for topic, group in topic_keywords.groupby("topic"):
        keywords = group["keyword"].dropna().astype(str).str.lower().str.strip()
        scores[str(topic).lower()] = sum(keyword in text for keyword in keywords)
    best_topic = max(scores, key=scores.get) if scores else None
    return best_topic if best_topic and scores[best_topic] > 0 else "feedback penonton"


def assign_realtime_category(row, category_keywords):
    topic = str(row.get("topic", "")).lower().strip()
    text = str(row.get("comment", "")).lower()
    tokens = set(text.split())
    labels = []

    df_kw = category_keywords.copy()
    df_kw["topic"] = df_kw["topic"].astype(str).str.lower().str.strip()
    df_kw = df_kw[df_kw["topic"] == topic]

    for _, kw_row in df_kw.iterrows():
        label = str(kw_row.get("label", "")).lower().strip()
        keyword = str(kw_row.get("keyword", "")).lower().strip()
        if not label or not keyword:
            continue
        if (" " in keyword and keyword in text) or (keyword in tokens):
            labels.append(label)

    return ", ".join(sorted(set(labels))) if labels else "feedback penonton"


def detect_realtime_sentiment(text, topic, sentiment_keywords):
    if not isinstance(text, str) or text.strip() == "":
        return None

    clean = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", str(text).lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    topic = str(topic).lower().strip()

    df_kw = sentiment_keywords.copy()
    df_kw["topic"] = df_kw["topic"].astype(str).str.lower().str.strip()
    df_kw["sentiment"] = df_kw["sentiment"].astype(str).str.lower().str.strip()
    df_kw["keyword"] = df_kw["keyword"].astype(str).str.lower().str.strip()
    topic_kw = df_kw[df_kw["topic"] == topic]
    if topic_kw.empty:
        topic_kw = df_kw

    pos_words = topic_kw[topic_kw["sentiment"].isin(["positive", "positif"])]["keyword"].tolist()
    neg_words = topic_kw[topic_kw["sentiment"].isin(["negative", "negatif"])]["keyword"].tolist()
    neu_words = topic_kw[topic_kw["sentiment"].isin(["neutral", "netral"])]["keyword"].tolist()

    rule_sentiment = None
    if any(word in clean for word in pos_words):
        rule_sentiment = "Positif"
    elif any(word in clean for word in neg_words):
        rule_sentiment = "Negatif"
    elif any(word in clean for word in neu_words):
        rule_sentiment = "Netral"

    polarity = TextBlob(clean).sentiment.polarity
    blob_sentiment = (
        "Positif" if polarity > 0.1
        else "Negatif" if polarity < -0.1
        else "Netral"
    )

    if rule_sentiment == blob_sentiment:
        return blob_sentiment
    if rule_sentiment and blob_sentiment != "Netral":
        return blob_sentiment
    if rule_sentiment and blob_sentiment == "Netral":
        return rule_sentiment
    return "Netral"


def prepare_realtime_sentiment(video_id, video_info, comments):
    kamus_data, topic_keywords, category_keywords, sentiment_keywords = load_realtime_keyword_data()
    df_live = pd.DataFrame(comments)
    if df_live.empty:
        return df_live

    df_live["video_id"] = video_id
    df_live["title"] = video_info["title"]
    df_live["channel"] = video_info["channel"]
    df_live["raw_comment"] = df_live["comment"]
    df_live["comment"] = df_live["comment"].apply(clean_realtime_text)
    df_live = df_live[df_live["comment"].str.strip() != ""].copy()
    if df_live.empty:
        return df_live

    df_live["case_folding"] = df_live["comment"].apply(case_fold_realtime_text)
    df_live["comment"] = df_live["case_folding"]
    df_live["tokenizing"] = df_live["case_folding"].apply(tokenize_realtime_text)
    df_live["normalization"] = df_live["tokenizing"].apply(
        lambda text: normalize_realtime_tokens(text, kamus_data)
    )
    stop_words = get_realtime_stopwords()
    df_live["stopword"] = df_live["normalization"].apply(
        lambda text: ", ".join([word for word in str(text).split(", ") if word.lower() not in stop_words])
    )
    df_live["stemming"] = df_live["stopword"].apply(simple_stem_text)
    df_live["topic"] = df_live["stemming"].apply(lambda text: identify_realtime_topic(text, topic_keywords))
    df_live["topik_result"] = df_live["topic"]
    df_live["category"] = df_live.apply(lambda row: assign_realtime_category(row, category_keywords), axis=1)
    df_live["sentiment"] = df_live.apply(
        lambda row: detect_realtime_sentiment(row["comment"], row["topic"], sentiment_keywords), axis=1
    )
    df_live["date"] = pd.to_datetime(df_live["date"], errors="coerce")
    df_live["year"] = df_live["date"].dt.year.fillna(pd.Timestamp.today().year).astype(int)
    df_live["date"] = df_live["date"].dt.date.astype(str)
    return df_live


def evaluate_realtime_video(df_live):
    default_metric = {
        "model_used": "SVM",
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
    }

    if df_live.empty or df_live["sentiment"].nunique() < 2 or df_live["sentiment"].value_counts().min() < 2:
        return default_metric

    X = df_live["stemming"].astype(str)
    y = df_live["sentiment"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(dual=False),
    }
    scores = {}
    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        scores[model_name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 2),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 2),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 2),
            "f1-score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 2),
        }

    best_name, best_metric = sorted(
        scores.items(), key=lambda item: (item[1]["f1-score"], item[1]["accuracy"]), reverse=True
    )[0]
    best_metric["model_used"] = best_name
    return best_metric


def summarize_realtime_video(df_live):
    counts = df_live["sentiment"].value_counts()
    pos = int(counts.get("Positif", 0))
    neu = int(counts.get("Netral", 0))
    neg = int(counts.get("Negatif", 0))
    total = pos + neu + neg
    return {
        "Positif": pos,
        "Netral": neu,
        "Negatif": neg,
        "Total": total,
        "Persentase_Positif (%)": f"{(pos / total * 100) if total else 0:.2f}%",
        "Persentase_Netral (%)": f"{(neu / total * 100) if total else 0:.2f}%",
        "Persentase_Negatif (%)": f"{(neg / total * 100) if total else 0:.2f}%",
    }


def save_realtime_to_database(df_live, metric):
    engine = get_postgres_engine()
    video_id = str(df_live["video_id"].iloc[0])
    summary = summarize_realtime_video(df_live)

    hasil_cols = [
        "video_id", "title", "channel", "author", "comment", "like_count", "date", "year",
        "normalization", "tokenizing", "stopword", "stemming", "category", "sentiment", "topic", "topik_result"
    ]
    df_hasil = df_live.reindex(columns=hasil_cols)
    df_mapping_live = df_live.reindex(columns=[
        "video_id", "author", "comment", "like_count", "date", "topic", "year",
        "normalization", "tokenizing", "stopword", "stemming", "category"
    ])
    df_best_live = df_live.copy()
    for key, value in metric.items():
        df_best_live[key] = value
    df_best_live = df_best_live.reindex(columns=[
        "video_id", "title", "channel", "author", "comment", "like_count", "date", "year",
        "normalization", "tokenizing", "stopword", "stemming", "category", "sentiment",
        "model_used", "accuracy", "precision", "recall", "f1-score", "topic", "topik_result"
    ])
    df_video_summary = pd.DataFrame([{
        "video_id": video_id,
        "Positif": summary["Positif"],
        "Netral": summary["Netral"],
        "Negatif": summary["Negatif"],
        "Persentase_Positif (%)": summary["Persentase_Positif (%)"],
        "Persentase_Netral (%)": summary["Persentase_Netral (%)"],
        "Persentase_Negatif (%)": summary["Persentase_Negatif (%)"],
        "Total": summary["Total"],
    }])

    with engine.begin() as conn:
        for table_name in ["hasil_sentimen", "mapping", "best_model_per_video", "sentiment_per_video"]:
            conn.execute(text(f"DELETE FROM {table_name} WHERE video_id = :video_id"), {"video_id": video_id})

    df_hasil.to_sql("hasil_sentimen", engine, if_exists="append", index=False)
    df_mapping_live.to_sql("mapping", engine, if_exists="append", index=False)
    df_best_live.to_sql("best_model_per_video", engine, if_exists="append", index=False)
    df_video_summary.to_sql("sentiment_per_video", engine, if_exists="append", index=False)

    df_all_sentiment = pd.read_sql_table("hasil_sentimen", engine)
    sentiment_per_topic = (
        df_all_sentiment.groupby("topic")["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["Positif", "Netral", "Negatif"]:
        if col not in sentiment_per_topic.columns:
            sentiment_per_topic[col] = 0
    sentiment_per_topic["Total"] = sentiment_per_topic["Positif"] + sentiment_per_topic["Netral"] + sentiment_per_topic["Negatif"]
    sentiment_per_topic["Persentase_Positif (%)"] = (sentiment_per_topic["Positif"] / sentiment_per_topic["Total"] * 100).fillna(0).round(2).astype(str) + "%"
    sentiment_per_topic["Persentase_Netral (%)"] = (sentiment_per_topic["Netral"] / sentiment_per_topic["Total"] * 100).fillna(0).round(2).astype(str) + "%"
    sentiment_per_topic["Persentase_Negatif (%)"] = (sentiment_per_topic["Negatif"] / sentiment_per_topic["Total"] * 100).fillna(0).round(2).astype(str) + "%"
    sentiment_per_topic = sentiment_per_topic[[
        "topic", "Positif", "Netral", "Negatif",
        "Persentase_Positif (%)", "Persentase_Netral (%)", "Persentase_Negatif (%)", "Total"
    ]]
    sentiment_per_topic.to_sql("sentiment_per_topik", engine, if_exists="replace", index=False)
    st.cache_data.clear()

# CSS DASHBOARD
st.markdown("""
<style>

/* TITLE DASHBOARD BESAR */
.page-title {
    font-size: 40px;
    font-weight: bold;
    color: white;
    margin-top: -110px;
    margin-bottom: -10px;
}

.page-subtitle {
    font-size: 14px;
    color: #dcfce7;
    font-weight: bold;
    margin-bottom: 20px;
}
/* KPI GLOBAL STYLE */
.kpi-global {
    display: flex;
    align-items: center;
    gap: 15px;               
    padding: 15px 16px;  
    min-height: 70px; 
    border-radius: 12px; 
    background: linear-gradient(135deg, #ffffff, #f0fdf4);
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    transition: 0.3s;
    margin-bottom: 10px;   
    margin-top: -25px;
}

.kpi-global:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

/* ICON */
.kpi-icon {
    font-size: 26px;    
    background: #064b22;
    color: white;
    padding: 12px;          
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* TITLE */
.kpi-title {
    font-size: 15px;
    color: #666;
    font-weight: 600;
    line-height: 1.3;
    margin-bottom: 2px;
}

/* VALUE */
.kpi-value {
    font-size: 23px;         
    font-weight: bold;
    color: #064b22;
    line-height: 1.2;
}

.timechart-header {
    box-sizing: border-box;
    background: rgba(5,54,22,0.95);
    padding: 14px 22px 10px;
    width: 100%;
    border: 1.5px solid rgba(34, 197, 94, 0.45);
    border-bottom: 0;
    border-radius: 12px 12px 0 0;
    display: block;
    margin: 2px 0 -10px;
    box-shadow: none;
}
            
.timechart-title {
    font-size: 18px;                
    font-weight: bold;               
    color: white;
    line-height: 1.2;
    text-align: center;
}

.timechart-subtitle {
    font-size: 13px;               
    color: white;
    margin-top: 4px;
    line-height: 1.2;
    text-align: center;
}

.element-container:has(.timechart-header) + .element-container {
    box-sizing: border-box;
    width: 100%;
    background: rgba(5,54,22,0.95);
    border: 1.5px solid rgba(34, 197, 94, 0.45);
    border-top: 1px solid rgba(52, 211, 153, 0.48);
    border-radius: 0 0 12px 12px;
    padding: 14px 22px 18px;
    margin: 0 0 -22px !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    backdrop-filter: blur(12px);
}

.element-container:has(.timechart-header) + .element-container div[data-baseweb="select"] {
    box-sizing: border-box !important;
    width: calc(100% - 50px) !important;
    margin-left: 2px !important;
    margin-right: auto !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.element-container:has(.timechart-header) + .element-container div[data-baseweb="select"] > div {
    box-sizing: border-box !important;
    width: 100% !important;
    max-width: 100% !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
    min-height: 38px !important;
    max-height: none !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

/* Mengatur Tinggi dan Ukuran Teks di dalam Box Multiselect / Selectbox */
div[data-baseweb="select"] > div {
    max-height: 30px !important; /* Tinggi kotak filter */
    font-size: 14px !important;  /* Ukuran teks pilihan di dalam box */
    border-radius: 10px !important;
}
            
/* Mengatur ukuran Chip/Tag yang muncul setelah dipilih (khusus multiselect) */
div[data-baseweb="tag"] {
    height: 10px !important;
    font-size: 12px !important;
    background-color: #0b7a3b !important; /* Warna hijau agar senada */
}

/* Mengatur ukuran teks Placeholder (Silakan pilih...) */
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    opacity: 0.8;
    color: #064b22;
}

/* Memberikan jarak antara box filter dan header di atasnya */
div[data-baseweb="select"] {
    margin-top: 5px !important; /* Menambah jarak agar tidak terlalu mepet ke atas */
    margin-bottom: 6px !important;
    border-radius: 0 0 10px 10px !important; 
}

/* Mengatur ukuran teks placeholder menjadi hijau pekat */
div[data-baseweb="select"] div[aria-live="polite"] {
    color: #064b22 !important;
    font-size: 11px !important;
}
                        
/* progress bar warna */
div[data-testid="stProgressBar"] > div > div {
    background-color: #16a34a;
    margin-top: 5px;
}

/* HEADER DALAM CARD */
.sentiment-header {
    background: linear-gradient(135deg, rgba(5,54,22,0.95), rgba(7,108,48,0.88));
    padding: 8px 12px;
    border-radius: 10px;
    border: 1.5px solid rgba(255,255,255,0.38);
    margin-bottom: 3px;
    margin-top: -15px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.15);
}

.sentiment-title {
    color: white;
    font-size: 18px;
    font-weight: bold;
    line-height: 1.2;
}

/* SUBTITLE */
.sentiment-subtitle {
    font-size: 12px;
    color: white;
    margin-top: 1px;
    line-height: 1.2;
}
            
/* CONTAINER STAT ITEM */
.stat-item {
    margin-bottom: 15px;
}

/* LABEL */
.sentiment-icon {
    width: 16px;
    height: 16px;
    margin-right: 8px;
    vertical-align: middle;
}

.stat-label {
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 1px;
}

/* VALUE */
.stat-value {
    font-size: 13px;
    color: #ffffff;
    margin-bottom: 2px;
    font-weight:600;
}

/* BAR BACKGROUND */
.stat-bar {
    width: 100%;
    height: 12px;
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
}

/* BAR FILL */
.stat-fill {
    height: 100%;
    border-radius: 10px;
}

/* WARNA */
.fill-pos { background: linear-gradient(90deg, #16a34a, #86efac); }
.fill-neu { background: linear-gradient(90deg, #FBC02D, #FFC107); }
.fill-neg { background: linear-gradient(90deg, #E53935, #EF5350); }
            
/* PROGRESS BAR */
div[data-testid="stProgressBar"] > div > div {
    background-color: #16a34a;
    border-radius: 10px;
}

/*  FILTER BOX */
.filter-box {
    margin-bottom: 10px;
}

/*  CHIP MERAH  */
div[data-baseweb="tag"] {
    background-color: #F44336 !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 500;
    padding: 2px 6px;
    margin-top:-50px;
    margin-bottom: 5px;
}

/* ICON X */
div[data-baseweb="tag"] svg {
    color: white !important;
}

/* HOVER EFFECT */
div[data-baseweb="tag"]:hover {
    background-color: #D32F2F !important;
    transform: scale(1.05);
}

/* BOX FILTER */
div[data-baseweb="select"] > div {
    background-color: #f5f5f5 !important;
    border-radius: 12px !important;
    padding: 0px;
    height: 20px;
    font-size: 20px;
    margin-top: -20px;
    margin-bottom: -30px;
}

/* RAPATKAN JARAK */
.element-container {
    margin-bottom: 0px !important;
}

div[data-baseweb="select"] {
    margin-bottom: -20px;
}

.visual-container {
    background: linear-gradient(135deg, rgba(5,54,22,0.95), rgba(7,108,48,0.88));
    padding: 8px 12px;
    border-radius: 10px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05);
    margin-top: 10px;   
    margin-bottom: 10px;
    border: 1.5px solid rgba(255,255,255,0.38);
}

/* BOX untuk masing-masing gambar */
.visual-box {
    background: rgba(255,255,255,0.95);
    border-radius: 14px;
    padding: 10px;
    height: 280px;            
    display: flex;
    align-items: center;
    justify-content: center;
}

/* WordCloud (kiri) */
.visual-img-wc {
    width: 100%;
    height: 260px;
    border-radius: 16px;
    object-fit: fill;
}

/* Frekuensi (kanan) */
.visual-img-bar {
    width: 100%;
    height: 260px;  
    object-fit: fill; 
    border-radius: 10px;
}

/* TEXT */
.visual-title {
    color: white;
    font-size: 18px;
    font-weight: bold;
    line-height: 1.2;
}

.visual-subtitle {
    font-size: 12px;
    color: white;
    margin-top: 1px;
    line-height: 1.2;
}

.sentiment-example-header {
    margin-bottom: 0 !important;
    border-radius: 10px 10px 0 0;
    border-bottom: 0;
}

.sentiment-example-select-anchor {
    display: none;
}

.element-container:has(.sentiment-example-select-anchor) + .element-container {
    margin-bottom: 0 !important;
}

.element-container:has(.sentiment-example-select-anchor) + .element-container div[data-baseweb="select"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.element-container:has(.sentiment-example-select-anchor) + .element-container div[data-baseweb="select"] > div {
    min-height: 30px !important;
    max-height: 35px !important;
    height: 35px !important;
    margin: 0 !important;
    padding: 0 14px !important;
    border-radius: 0 0 10px 10px !important;
    border: 0 !important;
    background-color: #f5f5f5 !important;
    box-shadow: 0 5px 12px rgba(0,0,0,0.10);
    display: flex !important;
    align-items: center !important;
    margin-bottom: -10px !important;
}

.element-container:has(.sentiment-example-select-anchor) + .element-container div[data-baseweb="select"] span,
.element-container:has(.sentiment-example-select-anchor) + .element-container div[data-baseweb="select"] div {
    font-size: 15px !important;
    font-weight: 500 !important;
    color: #5d5d5e !important;
}

.comment-example-card {
    background: rgba(255,255,255,0.96);
    border-left: 5px solid #16a34a;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 3px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

.comment-example-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 5px;
}

.comment-example-label {
    color: #064b22;
    font-size: 11px;
    font-weight: 900;
    text-transform: uppercase;
}

.comment-example-author {
    color: #d32f2f;
    font-size: 12px;
    font-weight: 800;
}

.comment-example-text {
    color: #263238;
    font-size: 13px;
    line-height: 1.45;
    font-weight: 500;
}

.comment-example-empty {
    background-color: rgba(255,255,255,0.1);
    padding: 12px;
    border-radius: 10px;
    border: 1px dashed rgba(255, 255, 255, 0.3);
    text-align: center;
    color: white;
    margin-top: 8px;
    font-size: 13px;
}

/* FILTER TITLE */
.filter-title {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: auto;
    padding: 14px 22px 10px;
    margin-top: -14px;
    margin-bottom: 0;
    color: #f7fff8;
    text-align: center;
    background: rgba(5,54,22,0.95);
    background-size: cover;
    background-position: center 54%;
    border: 1.5px solid rgba(34, 197, 94, 0.45);
    border-bottom: 0;
    border-radius: 12px 12px 0 0;
    box-shadow: none;
    position: relative;
    overflow: hidden;
}

.filter-title::before {
    display: none;
}

.filter-title-main {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    width: min(86%, 760px);
    position: relative;
    z-index: 2;
    margin-top: 0;
    color: #ffffff;
    font-size: 17px;
    font-weight: bold;
    line-height: 1.15;
    letter-spacing: 0;
    text-shadow: none;
}

.filter-title-main::before,
.filter-title-main::after {
    display: none;
}

.filter-title-main::after {
    display: none;
}

.filter-title-main + .filter-title-subtitle {
    position: relative;
    z-index: 2;
    width: 100%;
    margin-top: 4px;
    color: #e6f7ea;
    font-size: 12px;
    font-weight: 500;
    line-height: 1.35;
    text-align: center;
}

.element-container:has(.filter-title) + .element-container {
    box-sizing: border-box;
    width: 100%;
    background: rgba(5,54,22,0.95);
    border: 1.5px solid rgba(34, 197, 94, 0.45);
    border-top: 1px solid rgba(52, 211, 153, 0.48);
    border-radius: 0 0 12px 12px;
    padding: 14px 22px 18px;
    margin: 0 0 10px !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    backdrop-filter: blur(12px);
}

.element-container:has(.filter-title) + .element-container div[data-baseweb="select"] {
    box-sizing: border-box !important;
    width: calc(100% - 50px) !important;
    margin-left: 2px !important;
    margin-right: auto !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.element-container:has(.filter-title) + .element-container div[data-baseweb="select"] > div {
    box-sizing: border-box !important;
    width: 100% !important;
    max-width: 100% !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
    min-height: 38px !important;
    max-height: none !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.model-title {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    margin-top: 5px;
    margin-bottom: 5px;
    text-align: center;
}

.model-eval-filter-anchor {
    display: none;
}

.model-eval-table-anchor {
    display: none;
}

.model-eval-panel-title {
    box-sizing: border-box;
    width: min(100%, 1040px);
    padding: 18px 24px 12px;
    margin: -35px auto 0;
    background: rgba(5,54,22,0.95);
    border: 1.5px solid rgba(34, 197, 94, 0.45);
    border-bottom: 0;
    border-radius: 12px 12px 0 0;
    box-shadow: none;
}

.model-eval-panel-title .visual-title {
    font-size: 18px;
    text-align: center;
}

.model-eval-panel-title .visual-subtitle {
    margin-top: 4px;
    font-size: 12px;
    font-weight: 500;
    text-align: center;
    color: #e6f7ea;
}

.element-container:has(.model-eval-filter-anchor) + .element-container {
    margin-bottom: 0 !important;
}

.element-container:has(.model-eval-panel-title) + div[data-testid="stHorizontalBlock"] {
    box-sizing: border-box;
    width: min(100%, 1040px);
    background: rgba(5,54,22,0.95);
    border: 1.5px solid rgba(34, 197, 94, 0.45);
    border-top: 1px solid rgba(52, 211, 153, 0.48);
    border-radius: 0 0 12px 12px;
    padding: 18px 24px 20px;
    margin: 0 auto 10px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    backdrop-filter: blur(12px);
}

.model-eval-filter-label {
    color: #ffffff;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 5px;
}

.element-container:has(.model-eval-panel-title) + div[data-testid="stHorizontalBlock"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    min-height: 38px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.element-container:has(.model-eval-panel-title) + div[data-testid="stHorizontalBlock"] div[data-baseweb="select"] {
    margin-bottom: 0 !important;
}

.element-container:has(.model-eval-panel-title) + div[data-testid="stHorizontalBlock"] .element-container {
    margin-bottom: 0 !important;
}

.element-container:has(.model-eval-table-anchor) {
    margin-bottom: 0 !important;
}

.element-container:has(.model-eval-table-anchor) + .element-container {
    margin-top: -30px !important;
}

.model-eval-detail-title {
    margin-top: 0;
}

.element-container:has(.model-eval-detail-title) + .element-container {
    box-sizing: border-box;
    width: min(100%, 1040px);
    background: rgba(5,54,22,0.95);
    border: 1.5px solid rgba(34, 197, 94, 0.45);
    border-top: 1px solid rgba(52, 211, 153, 0.48);
    border-radius: 0 0 12px 12px;
    padding: 16px 24px 20px;
    margin: 0 auto 8px !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    backdrop-filter: blur(12px);
}

.element-container:has(.model-eval-detail-title) + .element-container div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    min-height: 38px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.element-container:has(.model-eval-detail-title) + .element-container div[data-baseweb="select"] {
    width: min(100%, 965px) !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    margin-bottom: 0 !important;
}

.model-eval-video-card {
    margin-top: -10px;
}

@media (max-width: 768px) {
    .model-eval-panel-title,
    .element-container:has(.model-eval-panel-title) + div[data-testid="stHorizontalBlock"],
    .element-container:has(.model-eval-detail-title) + .element-container {
        width: 100%;
        padding-left: 16px;
        padding-right: 16px;
    }
}
""", unsafe_allow_html=True)

if menu == "Dashboard":

    # ================= HEADER PAGE =================
    st.markdown("""
    <div class="page-title">Dashboard</div>
    <div class="page-subtitle">
    Ringkasan analisis sentimen komentar YouTube pada konten pertanian
    </div>
    """, unsafe_allow_html=True)

    # ================= KPI GLOBAL =================
    total_komentar_all = df_sentiment_video[['Positif','Netral','Negatif']].sum().sum()
    total_video = df_model['video_id'].nunique()
    total_topik = df_model['topic'].nunique()
    total_label = df_sentiment['category'].dropna().str.split(',').explode().str.strip().str.lower().nunique()

    g1, g2, g3, g4 = st.columns(4)

    def kpi_global(title, value, icon):
        st.markdown(f"""
        <div class="kpi-global">
            <div class="kpi-icon">{icon}</div>
            <div>
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with g1:
        kpi_global("Total Komentar", f"{int(total_komentar_all):,}", "💬")

    with g2:
        kpi_global("Jumlah Video", total_video, "🎥")

    with g3:
        kpi_global("Jumlah Topik", total_topik, "📚")

    with g4:
        kpi_global("Jumlah Label", total_label, "🏷️")

    # ================= TIME SERIES =================
    df_sentiment.columns = df_sentiment.columns.str.strip()
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'], errors='coerce')
    df_sentiment = df_sentiment.dropna(subset=['date'])
    df_year = df_sentiment.groupby(df_sentiment['date'].dt.year).size().reset_index(name='jumlah')
    df_year.rename(columns={'date': 'year'}, inplace=True)

    st.markdown("""
    <div class="timechart-header">
        <div class="timechart-title">
            Tren Distribusi Data Komentar
        </div>
        <div class="timechart-subtitle">
            Visualisasi jumlah komentar berdasarkan periode waktu
        </div>
    </div>
    """, unsafe_allow_html=True)

    tahun_list = sorted(df_year['year'].unique())

    selected_years = st.multiselect(
        label="Silakan pilih tahun",
        options=tahun_list,
        default=[], 
        placeholder="Silakan pilih tahun",
        label_visibility="collapsed"
    )

    if not selected_years:
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; border: 1px dashed rgba(255, 255, 255, 0.3);
                    text-align: center; color: white; font-size: 13px; margin-top: 10px; margin-bottom: 25px;">
            Sistem menunggu input tahun untuk menampilkan tren distribusi data komentar.
        </div>
        """, unsafe_allow_html=True)
    else:
        if len(selected_years) == 1:
            selected_year = selected_years[0]
            df_month = df_sentiment[df_sentiment['date'].dt.year == selected_year].copy()
            df_month['month'] = df_month['date'].dt.month
            df_filtered = df_month.groupby('month').size().reset_index(name='jumlah')
            all_months = pd.DataFrame({'month': range(1,13)})
            df_filtered = all_months.merge(df_filtered, on='month', how='left').fillna(0)
            x_col = 'month'
            x_label = "Bulan"
            x_ticks = range(1,13)
        else:
            df_filtered = df_year[df_year['year'].isin(selected_years)]
            x_col = 'year'
            x_label = "Tahun"
            x_ticks = df_filtered['year']

        fig, ax = plt.subplots(figsize=(7,2.1))
        ax.bar(df_filtered[x_col], df_filtered['jumlah'], color="#22c55e", edgecolor="#064b22", alpha=0.7)
        ax.plot(df_filtered[x_col], df_filtered['jumlah'], color="#E53935", marker='o', linewidth=2.0)

        for x, y in zip(df_filtered[x_col], df_filtered['jumlah']):
            ax.text(x, y + max(df_filtered['jumlah'])*0.03, str(int(y)), 
                    ha='center', fontsize=6, color="white", fontweight="bold")

        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#FFFFFF')
        ax.tick_params(colors='white', labelsize=7)
        ax.set_xlabel(x_label, color="white", fontweight="bold", fontsize=7)
        ax.set_ylabel("Jumlah Data", color="white", fontweight="bold", fontsize=7)
        ax.set_xticks(x_ticks)

        if x_col == 'month':
            ax.set_xticklabels(['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'])

        ax.grid(True, linestyle='-', alpha=0.3)
        plt.tight_layout()  
        st.pyplot(fig)

    # ================= KONFIGURASI FILTER TOPIK =================
    st.markdown(
        f'<div class="filter-title" style="--agri-header-bg: url(data:image/jpeg;base64,{bg_image});"><div class="filter-title-main">Konfigurasi Filter Topik</div><div class="filter-title-subtitle">Pilih topik untuk melihat analisis sentimen berdasarkan topik yang Anda inginkan</div></div>',
        unsafe_allow_html=True
    )
    
    topic_list = sorted(df_sentiment_topik['topic'].dropna().unique())
    all_options = ["Semua Topik"] + topic_list

    if 'selected_topics_state' not in st.session_state:
        st.session_state.selected_topics_state = []
    if 'prev_selected_topics' not in st.session_state:
        st.session_state.prev_selected_topics = []

    def handle_topics():
        current_selection = st.session_state.topic_filter
        if not st.session_state.selected_topics_state and current_selection:
            st.session_state.selected_topics_state = current_selection
            return
        if len(current_selection) > len(st.session_state.selected_topics_state):
            new_item = [x for x in current_selection if x not in st.session_state.selected_topics_state][0]
            if new_item == "Semua Topik":
                st.session_state.selected_topics_state = ["Semua Topik"]
            else:
                st.session_state.selected_topics_state = [x for x in current_selection if x != "Semua Topik"]
        else:
            st.session_state.selected_topics_state = current_selection

    selected_topics = st.multiselect(
        label="Silakan pilih topik",
        options=all_options,
        key="topic_filter", 
        on_change=handle_topics, 
        default=st.session_state.selected_topics_state,
        placeholder="Silakan pilih topik",
        label_visibility="collapsed"
    )
    st.session_state.prev_selected_topics = selected_topics

    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(5,54,22,0.95), rgba(7,108,48,0.88)); 
                    padding: 8px 12px; 
                    color: white; 
                    width: 100%;
                    margin-top: -15px; 
                    margin-bottom: -16px; 
                    border-radius: 10px; 
                    line-height: 1.2;
                    border: 1.5px solid rgba(255,255,255,0.38);
                    box-shadow: 0 3px 8px rgba(0,0,0,0.1);">
            <div style="font-size: 18px; font-weight: bold;">
                Konten dengan Komentar Terbanyak
            </div>
            <div style="font-size: 12px; margin-top: 1px; color: white; line-height: 1.2;">
                Daftar video yang memiliki tingkat interaksi tertinggi berdasarkan jumlah komentar pengguna
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not selected_topics:
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border: 1px dashed rgba(255, 255, 255, 0.3); text-align: center; color: white; margin-top: 5px; font-size: 13px;">
            Sistem menunggu input topik untuk menampilkan data interaksi.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        h1, h2 = st.columns([1.2, 1.8])
        with h1:
            st.markdown('<div class="sentiment-header"><div class="sentiment-title">Distribusi Sentimen</div><div class="sentiment-subtitle">Persentase sentimen analisis berdasarkan filter topik</div></div>', unsafe_allow_html=True)
        with h2:
            st.markdown('<div class="sentiment-header"><div class="sentiment-title">Statistik Sentimen</div><div class="sentiment-subtitle">Jumlah sentimen analisis berdasarkan filter topik</div></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border: 1px dashed rgba(255, 255, 255, 0.3); text-align: center; color: white; margin-top: 3px; font-size: 13px;">
            Sistem menunggu input topik untuk menampilkan distribusi dan statistik sentimen.
        </div>
        """, unsafe_allow_html=True)

        render_sentiment_comment_dropdown(
            pd.DataFrame(),
            key_prefix="dashboard_topic_empty",
            title="Contoh Kalimat Komentar Berdasarkan Sentimen",
            empty_message="Silakan pilih topik terlebih dahulu untuk menampilkan contoh komentar berdasarkan sentimen."
        )

    else:
        if "Semua Topik" in selected_topics:
            df_filtered_topic = df_sentiment_topik.copy()
            top_v_filter = df_sentiment_video.copy()
            label_info = "Semua Topik"
        else:
            df_filtered_topic = df_sentiment_topik[df_sentiment_topik['topic'].isin(selected_topics)]
            valid_video_ids = df_model[df_model['topic'].isin(selected_topics)]['video_id'].unique()
            top_v_filter = df_sentiment_video[df_sentiment_video['video_id'].isin(valid_video_ids)].copy()
            active_topics = selected_topics
            label_info = ", ".join(active_topics)

        # ================= KPI VIDEO (TERFILTER) =================
        if not top_v_filter.empty:
            top_v_filter['Total_Komentar'] = top_v_filter['Positif'] + top_v_filter['Netral'] + top_v_filter['Negatif']
            df_top_comments = top_v_filter.sort_values(by='Total_Komentar', ascending=False).head(3)
            
            t_col1, t_col2, t_col3 = st.columns(3)
            cols = [t_col1, t_col2, t_col3]
            
            for i, (idx, row) in enumerate(df_top_comments.iterrows()):
                v_id_top = row['video_id']
               
                video_data_match = df_sentiment[df_sentiment['video_id'] == v_id_top]
                
                if not video_data_match.empty:
                    title_top = video_data_match['title'].iloc[0]
                    channel_top = video_data_match['channel'].iloc[0] if 'channel' in video_data_match.columns else video_data_match['author'].iloc[0]
                else:
                    title_top = f"Video {v_id_top}"
                    channel_top = "Unknown Channel"
                
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: white; padding: 12px; border-radius: 12px; border-left: 5px solid #FFD700; margin-top: 8px; margin-bottom: 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); height: 140px; display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 11px; color: #666; background: #f1f1f1; padding: 2px 6px; border-radius: 4px; margin-bottom: 5px; font-family: monospace;">ID: {v_id_top}</div>
                        <div style="font-size: 13px; font-weight: bold; color: #064b22; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.2; margin-bottom: 3px;">
                            {title_top}
                        </div>
                        <div style="font-size: 11px; color: #d32f2f; font-weight: bold; margin-bottom: 8px;">
                            <span style="color: #666; font-weight: normal;">👤 Channel:</span> {channel_top}
                        </div>
                         <div style="font-size: 20px; font-weight: 900; color: #064b22;">
                            {int(row['Total_Komentar']):,} <span style="font-size: 12px; color: #666; font-weight: normal;">Komentar</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Tidak ada data video untuk topik ini.")

        # ================= SENTIMEN DAN STATISTIK ====================
        st.markdown("<br>", unsafe_allow_html=True)
        h1, h2 = st.columns([1.2, 1.8])

        with h1:
            st.markdown("""
            <div class="sentiment-header">
                <div class="sentiment-title">Distribusi Sentimen</div>
                <div class="sentiment-subtitle">Persentase sentimen analisis berdasarkan filter topik</div>
            </div>
            """, unsafe_allow_html=True)

        with h2:
            st.markdown("""
            <div class="sentiment-header">
                <div class="sentiment-title">Statistik Sentimen</div>
                <div class="sentiment-subtitle">Jumlah sentimen analisis berdasarkan filter topik</div>
            </div>
            """, unsafe_allow_html=True)

        pos = int(df_filtered_topic['Positif'].sum())
        neu = int(df_filtered_topic['Netral'].sum())
        neg = int(df_filtered_topic['Negatif'].sum())
        total = pos + neu + neg

        if total > 0:
            left, right = st.columns([1.2, 1.8])

            # ================= PIE CHART =================
            with left:
                df_pie = pd.DataFrame({
                    "Sentimen": ["Positif", "Netral", "Negatif"],
                    "Jumlah": [pos, neu, neg]
                })

                fig = px.pie(
                    df_pie,
                    names="Sentimen",
                    values="Jumlah",
                    color="Sentimen",
                    color_discrete_map={
                        "Positif": "#16a34a",
                        "Netral": "#FFC107",
                        "Negatif": "#F44336"
                    },
                )

                fig.update_traces(
                    textinfo="percent+label",
                    textfont=dict(size=10, color="black", family="Arial Black"),
                    hovertemplate="<b>%{label}</b><br>%{value:,} komentar<br>%{percent}"
                )

                fig.update_layout(
                    height=200,
                    margin=dict(t=10, b=10, l=10, r=10),
                    font=dict(size=10, family="Arial Black"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ================= STATISTIK =================
            with right:
                def stat_bar(label, value, total, css_class, icon_path):
                    percent = (value / total * 100) if total else 0
                    icon_base64 = get_base64_icon(icon_path)
                    st.markdown(f"""
                    <div class="stat-item">
                        <div class="stat-label">
                            <img src="data:image/png;base64,{icon_base64}" class="sentiment-icon">
                            {label}
                        </div>
                        <div class="stat-value">{value:,} komentar ({percent:.1f}%)</div>
                        <div class="stat-bar">
                            <div class="stat-fill {css_class}" style="width:{percent}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                            
                stat_bar("Sentimen Positif", pos, total, "fill-pos", "smile.png")
                stat_bar("Sentimen Netral", neu, total, "fill-neu", "neutral.png")
                stat_bar("Sentimen Negatif", neg, total, "fill-neg", "sad.png")

            # ================= INSIGHT / DOMINANT =================
            dominant_dict = {"Positif": pos, "Netral": neu, "Negatif": neg}
            sorted_sentiment = sorted(dominant_dict.items(), key=lambda x: x[1], reverse=True)
            top_label, top_value = sorted_sentiment[0]
            second_label, second_value = sorted_sentiment[1]
            df_topic_totals = df_filtered_topic.copy()
            df_topic_totals["Total_Komentar"] = (
                df_topic_totals["Positif"].fillna(0)
                + df_topic_totals["Netral"].fillna(0)
                + df_topic_totals["Negatif"].fillna(0)
            )
            topic_total_summary = ", ".join(
                f"{row['topic']} <b>{int(row['Total_Komentar']):,} komentar</b>"
                for _, row in df_topic_totals.sort_values("Total_Komentar", ascending=False).iterrows()
            )

            if "Semua Topik" in selected_topics:
                insight_text = (
                    f"Secara keseluruhan, mayoritas opini publik menunjukkan kecenderungan "
                    f"<b>sentimen {top_label.lower()}</b> dengan total <b>{top_value:,} komentar</b>, "
                    f"lebih tinggi dibanding sentimen <b>{second_label.lower()}</b> <b>{second_value:,}</b>. "
                    f"Total komentar keseluruhan adalah <b>{total:,} komentar</b>, dengan rincian per topik: "
                    f"{topic_total_summary}. Ini mengindikasikan banyak persepsi umum yang bersifat {top_label.lower()}."
                )
            else:
                insight_text = (
                    f"Pada topik <b>{label_info}</b> dengan total <b>{total:,} komentar</b>, analisis menunjukkan komentar didominasi "
                    f"<b>sentimen {top_label.lower()}</b> dengan jumlah <b>{top_value:,} komentar</b>, "
                    f"mengungguli sentimen <b>{second_label.lower()}</b> sebanyak <b>{second_value:,}</b>. "
                )

            st.markdown(f"""
            <div style="backdrop-filter: blur(10px); padding: 10px; border-radius: 12px; border-left: 6px solid #16a34a;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05); font-size: 14px; line-height: 1.6; color: #ffffff; margin-top: -5px;">
                {insight_text}
            </div>
            """, unsafe_allow_html=True)

            if "Semua Topik" in selected_topics:
                df_comment_examples = df_sentiment.copy()
            else:
                df_comment_examples = df_sentiment[df_sentiment["topic"].isin(selected_topics)].copy()
            render_sentiment_comment_dropdown(
                df_comment_examples,
                key_prefix="dashboard_topic",
                title="Contoh Kalimat Komentar Berdasarkan Sentimen"
            )

    # ================= VISUALISASI KATA ====================
    st.markdown("""
    <div class="visual-container">
        <div class="visual-title">Visualisasi Kata</div>
        <div class="visual-subtitle">
            WordCloud dan frekuensi kata berdasarkan topik yang dipilih
        </div>
    """, unsafe_allow_html=True)
        
    if not selected_topics:
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border: 1px dashed rgba(255, 255, 255, 0.3); text-align: center; color: white; margin-top: 1px; font-size: 13px;">
            Sistem menunggu input topik untuk menampilkan visualisasi kata dan WordCloud.
        </div>
        """, unsafe_allow_html=True)
    else:
        if "Semua Topik" in selected_topics:
            wc_file = "wordcloud_all.png"
            freq_file = "frekuensi_all.png"
        else:
            topic_name = re.sub(r'[^a-zA-Z0-9_]', '_', selected_topics[0].lower())
            wc_file = f"wordcloud_{topic_name}.png"
            freq_file = f"frekuensi_{topic_name}.png"

        col1, col2 = st.columns([1.4, 1.8])

        # WORDCLOUD
        with col1:
            encoded_wc = get_optional_base64_asset(wc_file)
            if encoded_wc:
                st.markdown(f"""
                <div class="visual-box">
                    <img src="data:image/png;base64,{encoded_wc}" class="visual-img-wc">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"File {wc_file} tidak ditemukan")

        # FREKUENSI
        with col2:
            encoded_freq = get_optional_base64_asset(freq_file)
            if encoded_freq:
                st.markdown(f"""
                <div class="visual-box">
                    <img src="data:image/png;base64,{encoded_freq}" class="visual-img-bar">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"File {freq_file} tidak ditemukan")
        
        # ================= INSIGHT KATA =================
        if "Semua Topik" in selected_topics:
            df_text = df_sentiment.copy()
            label_insight = "Semua Topik"
        else:
            df_text = df_sentiment[df_sentiment['topic'].isin(selected_topics)]
            label_insight = selected_topics[0]

        keyword_topic_filter = None if "Semua Topik" in selected_topics else selected_topics
        pos_words = ", ".join(get_sentiment_top_keywords(df_text, "Positif", topic_filter=keyword_topic_filter))
        neu_words = ", ".join(get_sentiment_top_keywords(df_text, "Netral", topic_filter=keyword_topic_filter))
        neg_words = ", ".join(get_sentiment_top_keywords(df_text, "Negatif", topic_filter=keyword_topic_filter))

        if "Semua Topik" in selected_topics:
            insight_sentimen = (
                f"Secara keseluruhan, kata yang paling dominan pada sentimen <b>positif</b> adalah "
                f"<b>{pos_words}</b>, sedangkan pada sentimen <b>netral</b> didominasi oleh "
                f"<b>{neu_words}</b>, dan sentimen <b>negatif</b> oleh <b>{neg_words}</b>. "
                f"Hal ini menunjukkan bahwa pola penggunaan kata mencerminkan karakteristik respon audiens."
            )
        else:
            insight_sentimen = (
                f"Pada topik <b>{label_insight}</b>, kata dominan pada sentimen <b>positif</b> adalah "
                f"<b>{pos_words}</b>, sementara sentimen <b>netral</b> didominasi oleh "
                f"<b>{neu_words}</b>, dan sentimen <b>negatif</b> oleh <b>{neg_words}</b>."
            )

        st.markdown(f"""
        <div style="backdrop-filter: blur(10px); padding: 10px; border-radius: 12px; border-left: 6px solid #16a34a;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05); font-size: 14px; line-height: 1.6; color: #ffffff; margin-top: 10px;">
            {insight_sentimen}
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# =================== SENTIMENT ANALYSIS ==================
# =========================================================
elif menu == "Sentiment Analysis":

    st.markdown("""
    <style>
    /* 1. CLEANUP UI DEFAULT */
    [data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0px;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}

    /* JUDUL HALAMAN */
    .page-sa-title {
        font-size: 36px;
        font-weight: bold;
        color: white;
        margin-top: -125px;
        margin-bottom: -5px;
    }

    .page-sa-subtitle {
        font-size: 14px;
        color: #dcfce7;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .video-filter-panel-anchor {
        display: none;
    }

    .filter-title-video {
        align-items: center;
        min-height: auto;
        box-sizing: border-box;
        width: min(100%, 1040px);
        padding: 14px 24px 12px;
        margin: -50px auto 0;
        background: rgba(5,54,22,0.95);
        border: 1.5px solid rgba(34, 197, 94, 0.45);
        border-bottom: 0;
        border-radius: 12px 12px 0 0;
        box-shadow: none;
        overflow: visible;
    }

    .filter-title-video::before {
        display: none;
    }

    .filter-title-video .filter-title-main {
        justify-content: center;
        width: 100%;
        gap: 0;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }

    .filter-title-video .filter-title-main::before {
        display: none;
    }

    .filter-title-video .filter-title-main::after {
        display: none;
    }

    .filter-title-video .filter-title-subtitle {
        position: relative;
        z-index: 2;
        width: 100%;
        margin: 4px auto 0;
        color: #e6f7ea;
        font-size: 12px;
        font-weight: 500;
        line-height: 1.35;
        text-align: center;
    }

    .element-container:has(.video-filter-panel-anchor) + .element-container {
        margin-bottom: 0 !important;
    }

    .element-container:has(.filter-title-video) + div[data-testid="stHorizontalBlock"] {
        box-sizing: border-box;
        width: min(100%, 1040px);
        background: rgba(5,54,22,0.95);
        border: 1.5px solid rgba(34, 197, 94, 0.45);
        border-top: 1px solid rgba(52, 211, 153, 0.48);
        border-radius: 0 0 12px 12px;
        padding: 14px 24px 18px;
        margin: 0 auto -15px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        backdrop-filter: blur(12px);
    }

    .video-filter-label {
        color: #ffffff;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 6px;
    }
                    
    /* STYLE SELECTBOX */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border-radius: 8px !important;
        min-height: 38px !important;
        margin-top: -18px !important;
        margin-bottom: -60px !important;
    }

    .element-container:has(.filter-title-video) + div[data-testid="stHorizontalBlock"] div[data-baseweb="select"] > div {
        min-height: 40px !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.12);
    }

    .element-container:has(.filter-title-video) + div[data-testid="stHorizontalBlock"] div[data-baseweb="select"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    /* KPI CARD STYLE */
    .kpi-box {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-bottom: 6px solid #064b22;
        margin-bottom: 25px;
    }
    .kpi-value { font-size: 22px; font-weight: bold; color: #064b22; }
    .kpi-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; margin-top: 5px; }

    /* PROFESIONAL EMPTY STATE */
    .empty-state-container {
        display: flex;
        align-items: center;
        gap: 14px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 15px 32px;
        text-align: left;
        margin-top: 8px;
    }
    .empty-state-icon {
        flex: 0 0 auto;
        color: #dcfce7;
        font-size: 24px;
        line-height: 1;
        opacity: 0.9;
    }
    .empty-state-copy { min-width: 0; }
    .empty-state-text { color: white; font-weight: 800; font-size: 15px; margin-bottom: 3px; }
    .empty-state-subtext { color: #dcfce7; font-size: 12px; }

    /* VIDEO CARD CONTAINER */
    .video-card-container-choose {
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-bottom: 5px solid #064b22;
        margin-bottom: 12px;
        overflow: hidden;
    }
    .video-card-content-choose { padding: 18px 20px 0px 20px; }
    .video-title-text-choose { font-size: 15px; font-weight: bold; color: #064b22; line-height: 1.4; }
    .video-channel-text-choose { font-size: 11px; color: #d32f2f; font-weight: bold; margin-top: 6px; }
    .video-id-text-choose { font-size: 10px; color: #666; background: #f1f1f1; padding: 2px 8px; border-radius: 5px; display: inline-block; margin-top: 8px; margin-bottom: 12px; }

    /* VIDEO CARD CONTAINER */
    .video-card-container {
        background: white;
        border-radius: 15px 15px 0px 0px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-bottom: 5px solid #064b22;
        margin-bottom: -50px;
        margin-top: -15px;
        overflow: hidden;
        height: 140px;
    }
    .video-card-content { padding: 18px 20px 0px 20px; }
    .video-title-text { font-size: 15px; font-weight: bold; color: #064b22; line-height: 1.4; }
    .video-channel-text { font-size: 11px; color: #d32f2f; font-weight: bold; margin-top: 6px; }
    .video-id-text { font-size: 10px; color: #666; background: #f1f1f1; padding: 2px 8px; border-radius: 5px; display: inline-block; margin-top: 8px; margin-bottom: 12px; }
    .video-comment-count { font-size: 10px; color: #ffffff; background: #064b22; padding: 2px 8px; border-radius: 5px; display: inline-block; margin-left: 6px; margin-top: 8px; margin-bottom: 12px; font-weight: 700; }

    /* TOMBOL ANALISIS & WATCH */
    .stButton > button {
        border-radius: 0px 0px 10px 10px !important;
        background-color: #064b22 !important;
        color: white !important;
        border: none !important;
        height: 40px;
        font-weight: bold;
        width: 100%;
    }
                
    div.stButton > button:hover {
        background-color: #16a34a !important; 
        color: white !important;
        font-weight: bold;
        transition: transform 0.2s ease-in-out;
    } 
                
    .btn-watch {
        display: inline-block;
        width: 100%;
        padding: 8px;
        background-color: #FF0000;
        color: white !important;
        text-decoration: none;
        font-weight: bold;
        border-radius: 10px;
        text-align: center;
        margin-bottom: -10px;
        font-size: 15px;
    }

    .btn-watch:hover {
        background-color: #ff3333 !important; 
        color: white !important;
        font-weight: bold;
        transition: transform 0.2s ease-in-out;
    } 
                
    /* KONTROL JARAK GLOBAL */
    .stVerticalBlock { gap: 0.4rem !important; }

    /* INSIGHT BOX STYLE */
    .insight-container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #16a34a;
        margin-top: 1px;
        margin-bottom: 6px;
        color: white;
        backdrop-filter: blur(10px);
    }
    .insight-title { font-weight: bold; font-size: 14px; margin-bottom: 5px; color: #dcfce7; }
    .insight-text { font-size: 14px; line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="page-sa-title">Sentiment Analysis</div>
        <div class="page-sa-subtitle">Analisis mendalam sentimen publik per konten video</div>
        """, unsafe_allow_html=True)

        @st.cache_data
        def load_mapping_data():
            return load_table_from_postgres("mapping")

        df_mapping = load_mapping_data()

        with st.container():
            st.markdown('<span class="video-filter-panel-anchor"></span>', unsafe_allow_html=True)
            st.markdown(
                f'''
                <div class="filter-title filter-title-video" style="--agri-header-bg: url(data:image/jpeg;base64,{bg_image});">
                    <div class="filter-title-main">Konfigurasi Filter Video</div>
                    <div class="filter-title-subtitle">Pilih topik dan kategori untuk menyesuaikan analisis sentimen sesuai kebutuhan Anda.</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown('<div class="video-filter-label">Topik</div>', unsafe_allow_html=True)
                topic_options = sorted(df_mapping['topic'].dropna().unique()) if not df_mapping.empty else []
                selected_topic = st.selectbox("Topic", options=topic_options, index=None, placeholder="Silakan Pilih Topik", key="sb_topic", label_visibility="collapsed")
            with f_col2:
                st.markdown('<div class="video-filter-label">Kategori</div>', unsafe_allow_html=True)
                category_options = []
                if selected_topic and not df_mapping.empty:
                    topic_data = df_mapping[df_mapping['topic'] == selected_topic]
                    category_options = topic_data['category'].value_counts().index.tolist()
                selected_category = st.selectbox("Category", options=category_options, index=None, placeholder="Silakan Pilih Kategori", key="sb_category", label_visibility="collapsed")

        if not selected_topic or not selected_category:
            st.markdown("""
            <div class="empty-state-container">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-copy">
                    <div class="empty-state-text">Mulai Analisis Sentimen</div>
                    <div class="empty-state-subtext">Gunakan filter di atas untuk melihat bagaimana audiens merespon suatu konten berdasarkan topik dan kategori yang spesifik.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif "selected_video" in st.session_state and st.session_state.selected_video:
            sel_vid = st.session_state.selected_video
            df_vid_stat = df_sentiment_video[df_sentiment_video['video_id'] == sel_vid]
            
            if not df_vid_stat.empty:
                video_detail_data = df_sentiment[df_sentiment['video_id'] == sel_vid]
                title_data_detail = video_detail_data['title']
                v_title_detail = title_data_detail.iloc[0] if not title_data_detail.empty else f"Video {sel_vid}"
                if not video_detail_data.empty and 'channel' in video_detail_data.columns:
                    v_channel_detail = video_detail_data['channel'].iloc[0]
                elif not video_detail_data.empty and 'author' in video_detail_data.columns:
                    v_channel_detail = video_detail_data['author'].iloc[0]
                else:
                    v_channel_detail = "Unknown Channel"
                
                st.markdown(f"""
                <div class="video-card-container-choose" style="border-bottom: 5px solid #064b22; border-radius: 15px; margin-top: 20px; margin-bottom: 10px;">
                    <div class="video-card-content-choose">
                        <div class="video-title-text-choose">{v_title_detail}</div>
                        <div class="video-channel-text-choose"><span style="color: #666; font-weight: normal;">Channel:</span> {v_channel_detail}</div>
                        <div class="video-id-text-choose">ID: {sel_vid}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ================= KPI CARDS SECTION =================
                pos, neu, neg = int(df_vid_stat['Positif'].sum()), int(df_vid_stat['Netral'].sum()), int(df_vid_stat['Negatif'].sum())
                total = pos + neu + neg
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1: st.markdown(f'<div class="kpi-box"><div class="kpi-value">{total:,}</div><div class="kpi-label">Total Komentar</div></div>', unsafe_allow_html=True)
                with kpi2: st.markdown(f'<div class="kpi-box"><div class="kpi-value" style="color:#16a34a">{pos:,}</div><div class="kpi-label">Sentimen Positif</div></div>', unsafe_allow_html=True)
                with kpi3: st.markdown(f'<div class="kpi-box"><div class="kpi-value" style="color:#FFC107">{neu:,}</div><div class="kpi-label">Sentimen Netral</div></div>', unsafe_allow_html=True)
                with kpi4: st.markdown(f'<div class="kpi-box"><div class="kpi-value" style="color:#F44336">{neg:,}</div><div class="kpi-label">Sentimen Negatif</div></div>', unsafe_allow_html=True)

                # ============== Row 1: Chart Headers ================
                h1, h2 = st.columns([1.2, 1.8])
                with h1: st.markdown('<div class="sentiment-header"><div class="sentiment-title">Distribusi Sentimen</div><div class="sentiment-subtitle">Persentase sentimen berdasarkan video yang dipilih</div></div>', unsafe_allow_html=True)
                with h2: st.markdown('<div class="sentiment-header"><div class="sentiment-title">Statistik Sentimen</div><div class="sentiment-subtitle">Jumlah Komentar berdasarkan video yang dipilih</div></div>', unsafe_allow_html=True)

                st.markdown('<div style="margin-top: 28px; margin-bottom: -80px;"></div>', unsafe_allow_html=True)

                # ============== Row 2: Charts ================
                c_chart1, c_chart2 = st.columns([1.2, 1.8])
                with c_chart1:
                    fig = px.pie(values=[pos, neu, neg], names=["Positif", "Netral", "Negatif"], color=["Positif", "Netral", "Negatif"], color_discrete_map={"Positif": "#16a34a", "Netral": "#FFC107", "Negatif": "#F44336"})
                    fig.update_traces(textinfo="percent+label", textfont=dict(size=10, color="black", family="Arial Black"))
                    fig.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with c_chart2:
                    def render_stat(label, value, total, css_class, icon_path):
                        percent = (value / total * 100) if total else 0
                        icon_base64 = get_base64_icon(icon_path)
                        st.markdown(f"""<div class="stat-item"><div class="stat-label"><img src="data:image/png;base64,{icon_base64}" style="width:16px;"> {label}</div><div class="stat-value">{value:,} komentar ({percent:.1f}%)</div><div class="stat-bar"><div class="stat-fill {css_class}" style="width:{percent}%"></div></div></div>""", unsafe_allow_html=True)
                    render_stat("Positif", pos, total, "fill-pos", "smile.png")
                    render_stat("Netral", neu, total, "fill-neu", "neutral.png")
                    render_stat("Negatif", neg, total, "fill-neg", "sad.png")

                # --- INSIGHT CHART ---
                sentimen_dict = {"Positif": pos, "Netral": neu, "Negatif": neg}
                dominant_sentiment = max(sentimen_dict, key=sentimen_dict.get)
                persentase_dominan = (sentimen_dict[dominant_sentiment] / total * 100) if total > 0 else 0

                if dominant_sentiment == "Positif":
                    insight_desc = f"audiens memberikan respon yang sangat baik. Hal ini menunjukkan bahwa konten video tersebut relevan, informatif, atau menghibur bagi penonton."
                elif dominant_sentiment == "Netral":
                    insight_desc = f"audiens cenderung memberikan respon yang objektif atau bisa berarti bahwa konten tersebut dianggap biasa saja, tidak menimbulkan reaksi yang signifikan baik positif maupun negatif."
                else: 
                    insight_desc = f"terdapat cukup banyak kritik atau ketidakpuasan dari audiens. Hal ini bisa menjadi indikasi bahwa konten video tersebut mungkin kontroversial, kurang relevan, atau menimbulkan reaksi negatif di kalangan penonton."

                st.markdown(f"""
                <div class="insight-container">
                    <div class="insight-text">
                        Berdasarkan data yang dianalisis, video dengan judul <b>{v_title_detail}</b> memiliki sentimen dominan sebesar <b>{persentase_dominan:.1f}%</b> yang mengarah ke <b>{dominant_sentiment}</b>. 
                        Secara keseluruhan, ini menunjukkan bahwa {insight_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                render_sentiment_comment_dropdown(
                    video_detail_data,
                    key_prefix=f"sentiment_video_{sel_vid}",
                    title="Contoh Kalimat Komentar Berdasarkan Sentimen"
                )

                st.markdown("""<div class="visual-container"><div class="visual-title">Visualisasi Kata</div><div class="visual-subtitle">WordCloud dan frekuensi kata berdasarkan video yang dipilih</div></div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                v_col1, v_col2 = st.columns([1.4, 1.8])
                wc_file = f"wordcloud_{sel_vid}.png".lower()
                freq_file = f"frekuensi_{sel_vid}.png".lower()

                with v_col1:
                    encoded_wc = get_optional_base64_asset(wc_file)
                    if encoded_wc:
                        st.markdown(f'<div class="visual-box" style="border-radius:15px; margin-top: -28px; margin-bottom: 6px; padding:30px;"><img src="data:image/png;base64,{encoded_wc}" class="visual-img"></div>', unsafe_allow_html=True)
                with v_col2:
                    encoded_freq = get_optional_base64_asset(freq_file)
                    if encoded_freq:
                        st.markdown(f'<div class="visual-box" style="border-radius:15px; margin-top: -28px; margin-bottom: 6px; padding:15px;"><img src="data:image/png;base64,{encoded_freq}" class="visual-img"></div>', unsafe_allow_html=True)

                df_video_keywords = df_sentiment[df_sentiment["video_id"] == sel_vid].copy()
                video_topic_filter = (
                    df_video_keywords["topic"].dropna().astype(str).str.strip().unique().tolist()
                    if "topic" in df_video_keywords.columns else None
                )
                pos_words = ", ".join(get_sentiment_top_keywords(df_video_keywords, "Positif", topic_filter=video_topic_filter))
                neu_words = ", ".join(get_sentiment_top_keywords(df_video_keywords, "Netral", topic_filter=video_topic_filter))
                neg_words = ", ".join(get_sentiment_top_keywords(df_video_keywords, "Negatif", topic_filter=video_topic_filter))

                # Insight Kata Kunci
                st.markdown(f"""
                <div style="backdrop-filter: blur(10px); padding: 15px; border-radius: 12px; border-left: 6px solid #16a34a;
                    background-color: rgba(255, 255, 255, 0.1); box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
                    font-size: 14px; line-height: 1.6; color: #ffffff;">
                    Pada video <b>{v_title_detail}</b>, kata-kata yang paling sering muncul pada respon <b>positif</b> adalah 
                    <span style="color: #ffffff;"><b>{pos_words}</b></span>. Sementara itu, komentar <b>netral</b> banyak menggunakan kata 
                    <span style="color: #ffffff;"><b>{neu_words}</b></span>, dan pada sisi <b>negatif</b> didominasi oleh penggunaan kata 
                    <span style="color: #ffffff;"><b>{neg_words}</b></span>. Hal ini menggambarkan fokus utama audiens dalam menanggapi video tersebut.
                </div>
                """, unsafe_allow_html=True)

                # Action Buttons 
                st.markdown("<br>", unsafe_allow_html=True)
                yt_url = f"https://www.youtube.com/watch?v={sel_vid}"
                st.markdown(f'<a href="{yt_url}" target="_blank" class="btn-watch">Lihat Video</a>', unsafe_allow_html=True)
                
                if st.button("Kembali ke Daftar Video", key="btn_back_to_list", use_container_width=True):
                    st.session_state.selected_video = None
                    st.rerun()

        # DAFTAR CARD (Berdasarkan Filter)
        else:
            filtered_ids = df_mapping[(df_mapping['topic'] == selected_topic) & (df_mapping['category'] == selected_category)]['video_id'].unique()
            df_display = df_model[df_model['video_id'].isin(filtered_ids)].drop_duplicates(subset=['video_id'])
            df_comment_totals = df_sentiment_video[df_sentiment_video['video_id'].isin(filtered_ids)].copy()
            if not df_comment_totals.empty:
                df_comment_totals['Total_Komentar'] = (
                    df_comment_totals['Positif'].fillna(0)
                    + df_comment_totals['Netral'].fillna(0)
                    + df_comment_totals['Negatif'].fillna(0)
                )
                df_comment_totals = df_comment_totals[['video_id', 'Total_Komentar']].groupby('video_id', as_index=False).sum()
                df_display = df_display.merge(df_comment_totals, on='video_id', how='left')
            else:
                df_display['Total_Komentar'] = 0
            df_display['Total_Komentar'] = df_display['Total_Komentar'].fillna(0).astype(int)
            df_display = df_display.sort_values(by='Total_Komentar', ascending=False)
            st.markdown(f"<div style='color:white; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(6px); border-radius: 8px; padding: 8px; padding-left: 20px; font-weight:bold; margin-bottom:20px; margin-top: 20px; font-size:16px;'>Hasil pencarian dari topik {selected_topic} dengan kategori {selected_category} ada {len(df_display)} video</div>", unsafe_allow_html=True)

            if df_display.empty:
                st.warning("Tidak ditemukan video untuk kombinasi topik dan kategori ini.")
            else:
                for idx, row in df_display.iterrows():
                    v_id = str(row['video_id'])
                    total_comments = int(row.get('Total_Komentar', 0))
                    video_data = df_sentiment[df_sentiment['video_id'] == v_id]
                    title_data = video_data['title']
                    v_title = title_data.iloc[0] if not title_data.empty else f"Video {v_id}"
                    if not video_data.empty and 'channel' in video_data.columns:
                        v_channel = video_data['channel'].iloc[0]
                    elif not video_data.empty and 'author' in video_data.columns:
                        v_channel = video_data['author'].iloc[0]
                    else:
                        v_channel = "Unknown Channel"
                    st.markdown(f"""<div class="video-card-container"><div class="video-card-content"><div class="video-title-text">{v_title}</div><div class="video-channel-text"><span style="color: #666; font-weight: normal;">Channel:</span> {v_channel}</div><div class="video-id-text">ID: {v_id}</div><div class="video-comment-count">{total_comments:,} Komentar</div></div>""", unsafe_allow_html=True)
                    if st.button("Lihat Analisis Lengkap", key=f"btn_ana_{v_id}_{idx}", use_container_width=True):
                        st.session_state.selected_video = v_id
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            
# =========================================================
# =================== REALTIME ANALYSIS ===================
# =========================================================
elif menu == "Realtime Analysis":

    st.markdown("""
    <style>
    [data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0px;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .page-realtime-title {
        font-size: 32px;
        font-weight: bold;
        color: white;
        margin-top: -105px;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    .page-realtime-subtitle {
        font-size: 14px;
        color: #dcfce7;
        font-weight: bold;
        margin-bottom: -10px;
        line-height: 1.4;
    }
    .realtime-panel-anchor {
        display: none;
    }
    div[data-testid="stVerticalBlock"]:has(> .element-container .realtime-panel-anchor) {
        box-sizing: border-box;
        width: 100%;
        max-width: none;
        margin: -25px 0 10px;
        padding: 0 30px 14px;
        background: rgba(5,54,22,0.95);
        border: 1.5px solid rgba(34, 197, 94, 0.45);
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.24);
        backdrop-filter: blur(12px);
        overflow: hidden;
    }
    .realtime-process-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0;
        box-sizing: border-box;
        width: calc(100% + 60px);
        min-height: auto;
        padding: 18px 30px 12px;
        color: #f7fff8;
        text-align: center;
        margin: -20px -30px 15px;
        background: transparent;
        border: 0;
        border-bottom: 0;
        border-radius: 0;
        box-shadow: none;
        position: relative;
        overflow: hidden;
    }
    .realtime-process-header::after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        height: 1px;
        background: rgba(52, 211, 153, 0.48);
    }
    .realtime-result-shell {
        box-sizing: border-box;
        width: 100%;
        margin-top: -35px;
        margin-bottom: 16px;
        padding: 18px 22px 20px;
        background: rgba(5,54,22,0.95);
        border: 1.5px solid rgba(34, 197, 94, 0.48);
        border-radius: 14px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(14px);
        overflow: hidden;
    }
    .realtime-result-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: calc(100% + 44px);
        min-height: auto;
        padding: 0 22px 12px;
        color: #f7fff8;
        text-align: center;
        margin: 6px -22px 14px;
        background: transparent;
        border: 0;
        border-bottom: 1px solid rgba(52, 211, 153, 0.32);
        border-radius: 0;
        box-shadow: none;
        position: relative;
        overflow: hidden;
    }
    .realtime-result-title {
        position: relative;
        z-index: 2;
        color: #ffffff;
        font-size: 18px;
        font-weight: 900;
        line-height: 1.15;
        margin-top: -2px;
        text-align: center;
    }
    .realtime-result-subtitle {
        position: relative;
        z-index: 2;
        color: #e6f7ea;
        font-size: 12px;
        font-weight: 500;
        line-height: 1.35;
        margin-top: 5px;
        text-align: center
    }
    .realtime-process-header::before,
    .realtime-result-header::before {
        display: none;
    }
    .realtime-process-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        width: 100%;
        position: relative;
        z-index: 2;
        margin-top: 0;
        color: #ffffff;
        font-size: 19px;
        font-weight: 900;
        line-height: 1.15;
        letter-spacing: 0;
        text-align: center;
        text-shadow: none;
    }
    .realtime-process-subtitle {
        position: relative;
        z-index: 2;
        color: #e6f7ea;
        font-size: 12px;
        font-weight: 500;
        line-height: 1.35;
        text-align: center;
        margin-top: 3px;
    }
    .realtime-process-copy {
        width: 100%;
        min-width: 0;
        transform: translateX(-28px);
    }
    .realtime-process-title::before,
    .realtime-process-title::after {
        display: none;
    }
    .realtime-panel {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        margin-bottom: 0;
        box-shadow: none;
    }
    .realtime-input-label {
        color: #ffffff;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: -30px;
    }
    .realtime-field-label {
        color: #ffffff;
        font-size: 14px;
        font-weight: 800;
        margin: 10px 0 -25px;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container,
    .element-container:has(.realtime-input-anchor) + .element-container,
    .element-container:has(.realtime-analyze-anchor) + .element-container {
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow: visible !important;
    }
    .element-container:has(.realtime-mode-anchor),
    .element-container:has(.realtime-analyze-anchor),
    .element-container:has(.realtime-input-anchor) {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] [role="radiogroup"],
    .element-container:has(.realtime-mode-anchor) + .element-container [role="radiogroup"] {
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 10px;
        margin-bottom: -15px !important;
        align-items: stretch !important;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] [role="radiogroup"] label,
    .element-container:has(.realtime-mode-anchor) + .element-container label {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.28) !important;
        border-radius: 8px !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        padding-left: 14px !important;
        padding-right: 14px !important;
        color: #ffffff !important;
        font-size: 12px !important;
        font-weight: 900 !important;
        box-shadow: none !important;
        box-sizing: border-box !important;
        flex: 0 0 calc((100% - 30px) / 2) !important;
        min-width: 0 !important;
        max-width: calc((100% - 30px) / 2) !important;
        min-height: 42px !important;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] {
        margin-bottom: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] > label {
        display: none !important;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] > div,
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] > div > div {
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] [role="radiogroup"] label > div:not(:first-child) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] [role="radiogroup"] label p,
    .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] [role="radiogroup"] label span,
    .element-container:has(.realtime-mode-anchor) + .element-container label p,
    .element-container:has(.realtime-mode-anchor) + .element-container label span {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 12px !important;
    }
    .element-container:has(.realtime-input-anchor) + .element-container div[data-testid="stTextInput"] {
        margin-top: 0 !important;
        margin-bottom: 6px !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    .element-container:has(.realtime-input-anchor) + .element-container div[data-testid="stTextInput"] > div {
        margin-bottom: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    .element-container:has(.realtime-input-anchor) + .element-container div[data-testid="stTextInput"] div[data-baseweb="input"],
    .element-container:has(.realtime-input-anchor) + .element-container div[data-testid="stTextInput"] input {
        box-sizing: border-box !important;
        width: 100% !important;
        min-height: 34px !important;
        border-radius: 8px !important;
    }
    .element-container:has(.realtime-input-anchor) + .element-container div[data-testid="stTextInput"] input {
        min-height: 36px !important;
        padding-left: 14px !important;
        padding-right: 14px !important;
        font-size: 13px !important;
    }
    .realtime-input-anchor,
    .realtime-analyze-anchor {
        display: block;
        height: 0;
        margin: 0;
        padding: 0;
    }
    div[data-testid="stSpinner"],
    div[data-testid="stSpinner"] *,
    div[data-testid="stSpinner"] p {
        color: #ffffff !important;
    }
    .element-container:has(.realtime-analyze-anchor) + .element-container .stButton > button {
        background-color: #087333 !important;
        color: #ffffff !important;
        border: 1.5px solid rgba(220, 252, 231, 0.72) !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        box-sizing: border-box !important;
        width: 100% !important;
        min-height: 42px !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12), 0 5px 12px rgba(0,0,0,0.18) !important;
        margin-top: -50px !important;
        margin-bottom: 5px !important;
        font-size: 14px !important;
        transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
    }
    .element-container:has(.realtime-analyze-anchor) + .element-container .stButton {
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow: visible !important;
    }
    .element-container:has(.realtime-analyze-anchor) + .element-container {
        margin-top: -20px !important;
        margin-bottom: 0 !important;
    }
    .element-container:has(.realtime-analyze-anchor) + .element-container .stButton > button:hover {
        background-color: #0aa34d !important;
        border-color: rgba(220, 252, 231, 0.95) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.18), 0 9px 18px rgba(0,0,0,0.26) !important;
    }
    .realtime-video-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(8, 68, 36, 0.92), rgba(4, 35, 22, 0.96));
        border-radius: 14px;
        border: 1px solid rgba(52, 211, 153, 0.32);
        padding: 14px 16px;
        margin: 0;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 12px 28px rgba(0,0,0,0.22);
        color: #ffffff;
    }
    .realtime-video-card::after {
        content: "";
        position: absolute;
        right: 36px;
        top: 22px;
        width: 118px;
        height: 78px;
        opacity: 0.16;
        pointer-events: none;
    }
    .realtime-video-content {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
    }
    .realtime-video-mark {
        flex: 0 0 auto;
        width: 58px;
        height: 58px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.26), rgba(5, 54, 22, 0.72));
        color: #bbf7d0;
        font-size: 28px;
        font-weight: 900;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .realtime-video-info {
        min-width: 0;
        flex: 1 1 auto;
    }
    .realtime-video-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: 900;
        margin-bottom: 6px;
        line-height: 1.2;
        text-transform: uppercase;
        overflow-wrap: anywhere;
    }
    .realtime-meta {
        font-size: 12px;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.35;
    }
    .realtime-chip {
        font-size: 11px;
        color: #fff;
        background: rgba(22, 163, 74, 0.38);
        border: 1px solid rgba(34, 197, 94, 0.45);
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 800;
        margin-right: 8px;
    }
    .realtime-chip-light {
        font-size: 11px;
        color: #f8fafc;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        margin-right: 8px;
    }
    .realtime-kpi {
        background: white;
        padding: 13px;
        border-radius: 12px;
        text-align: center;
        border-bottom: 6px solid #064b22;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }
    .realtime-kpi-value {
        font-size: 20px;
        font-weight: 900;
        color: #064b22;
    }
    .realtime-kpi-label {
        font-size: 10px;
        color: #666;
        font-weight: 800;
        text-transform: uppercase;
    }
    .realtime-empty-state {
        display: flex;
        align-items: center;
        gap: 14px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 15px 32px;
        text-align: left;
        margin-top: -20px;
    }
    .realtime-empty-icon {
        flex: 0 0 auto;
        color: #dcfce7;
        font-size: 24px;
        line-height: 1;
    }
    .realtime-empty-copy {
        min-width: 0;
    }
    .realtime-empty-title { color: white; font-weight: 800; font-size: 15px; margin-bottom: 3px; }
    .realtime-empty-subtitle { color: #dcfce7; font-size: 12px; }
    .realtime-insight {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 12px;
        border-radius: 12px;
        border-left: 5px solid #16a34a;
        margin-top: 2px;
        margin-bottom: 12px;
        color: white;
        font-size: 14px;
        line-height: 1.5;
    }
    .realtime-watch {
        display: inline-block;
        width: 100%;
        padding: 9px;
        background-color: #FF0000;
        color: white !important;
        text-decoration: none;
        font-weight: bold;
        border-radius: 10px;
        text-align: center;
        font-size: 15px;
    }
    .realtime-watch:hover { background-color: #ff3333 !important; color: white !important; }
    .realtime-wordcloud-img {
        width: 100%;
        height: 260px;
        object-fit: contain;
        border-radius: 10px;
    }
    .realtime-sentiment-gap {
        height: 14px;
    }
    @media (max-width: 900px) {
        div[data-testid="stVerticalBlock"]:has(> .element-container .realtime-panel-anchor) {
            padding: 18px 16px 20px;
        }
        .realtime-process-title {
            font-size: 18px;
        }
        .realtime-process-subtitle {
            font-size: 12px;
        }
        .realtime-input-label {
            font-size: 14px;
        }
        .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] [role="radiogroup"],
        .element-container:has(.realtime-mode-anchor) + .element-container [role="radiogroup"] {
            gap: 12px;
        }
        .element-container:has(.realtime-mode-anchor) + .element-container div[data-testid="stRadio"] [role="radiogroup"] label,
        .element-container:has(.realtime-mode-anchor) + .element-container label {
            flex-basis: calc((100% - 12px) / 2) !important;
            max-width: calc((100% - 12px) / 2) !important;
            min-height: 44px !important;
            padding: 10px 12px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-realtime-title">Realtime Sentiment Analysis</div>
    <div class="page-realtime-subtitle">Analisis sentimen komentar YouTube secara langsung berdasarkan ID video atau link video</div>
    """, unsafe_allow_html=True)

    if "realtime_result" not in st.session_state:
        st.session_state.realtime_result = None
    if "realtime_metric" not in st.session_state:
        st.session_state.realtime_metric = None
    if "realtime_video_id" not in st.session_state:
        st.session_state.realtime_video_id = ""

    df_live = st.session_state.realtime_result
    metric_live = st.session_state.realtime_metric
    video_id_live = st.session_state.realtime_video_id
    realtime_input_error = False
    realtime_error_message = None

    if df_live is None or df_live.empty:
        with st.container():
            st.markdown('<span class="realtime-panel-anchor"></span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="realtime-process-header" style="--agri-header-bg: url(data:image/jpeg;base64,{bg_image});">
                <div class="realtime-process-copy">
                    <div class="realtime-process-title">Proses Analisis Realtime</div>
                    <div class="realtime-process-subtitle">Masukkan link YouTube atau ID video untuk memulai analisis sentimen secara otomatis.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="realtime-input-label">Pilih Metode Input</div>', unsafe_allow_html=True)
            st.markdown('<span class="realtime-mode-anchor"></span>', unsafe_allow_html=True)
            realtime_input_mode = st.radio(
                "Pilih Metode Input",
                options=["By Link Video", "By Video ID"],
                horizontal=True,
                label_visibility="collapsed",
                key="realtime_input_mode"
            )
            input_placeholder = (
                "Masukkan link YouTube (contoh: https://www.youtube.com/watch?v=xxxxxxx)"
                if realtime_input_mode == "By Link Video"
                else "Masukkan Video ID (11 karakter YouTube)"
            )
            input_label = "Link YouTube" if realtime_input_mode == "By Link Video" else "Video ID"
            st.markdown(f'<div class="realtime-field-label">{input_label}</div>', unsafe_allow_html=True)
            st.markdown('<span class="realtime-input-anchor"></span>', unsafe_allow_html=True)
            realtime_input = st.text_input(
                "Masukkan Link YouTube" if realtime_input_mode == "By Link Video" else "Masukkan Video ID",
                placeholder=input_placeholder,
                label_visibility="collapsed"
            )
            st.markdown('<span class="realtime-analyze-anchor"></span>', unsafe_allow_html=True)
            analyze_clicked = st.button("Analisis Sekarang", key="btn_realtime_analyze", use_container_width=True)

        if analyze_clicked:
            if realtime_input_mode == "By Video ID":
                video_id = str(realtime_input or "").strip()
            else:
                video_id = extract_youtube_video_id(realtime_input)
            if not video_id:
                realtime_input_error = True
                realtime_error_message = "Link YouTube tidak valid." if realtime_input_mode == "By Link Video" else "Video ID tidak boleh kosong."
            elif not re.fullmatch(r"[\w-]{11}", video_id):
                realtime_input_error = True
                realtime_error_message = "Video ID harus terdiri dari 11 karakter YouTube yang valid."
            else:
                try:
                    with st.spinner("Mengambil semua komentar dan menjalankan analisis sentimen..."):
                        video_info = fetch_youtube_video(video_id)
                        comments = fetch_youtube_comments(video_id)
                        if not comments:
                            st.warning("Komentar tidak ditemukan atau komentar pada video ini tidak tersedia.")
                        else:
                            df_live = prepare_realtime_sentiment(video_id, video_info, comments)
                            metric_live = evaluate_realtime_video(df_live)
                            st.session_state.realtime_result = df_live
                            st.session_state.realtime_metric = metric_live
                            st.session_state.realtime_video_id = video_id
                            st.rerun()
                except Exception as exc:
                    realtime_input_error = True
                    realtime_error_message = f"Analisis gagal: {exc}"

        if realtime_error_message:
            st.error(realtime_error_message)

    if df_live is not None and not df_live.empty:
        summary_live = summarize_realtime_video(df_live)
        title_live = df_live["title"].iloc[0]
        channel_live = df_live["channel"].iloc[0]
        pos_live = summary_live["Positif"]
        neu_live = summary_live["Netral"]
        neg_live = summary_live["Negatif"]
        total_live = summary_live["Total"]
        title_live_safe = html.escape(str(title_live))
        channel_live_safe = html.escape(str(channel_live))
        video_id_live_safe = html.escape(str(video_id_live))
            
        st.markdown(f"""
        <div class="realtime-result-shell">
            <div class="realtime-result-header" style="--agri-header-bg: url(data:image/jpeg;base64,{bg_image});">
                <div class="realtime-result-title">Hasil Analisis Realtime</div>
                <div class="realtime-result-subtitle">Ringkasan hasil analisis sentimen dari komentar pada video yang dipilih secara realtime.</div>
            </div>
            <div class="realtime-video-card">
                <div class="realtime-video-content">
                    <div class="realtime-video-mark">🌱</div>
                    <div class="realtime-video-info">
                        <div class="realtime-video-title">{title_live_safe}</div>
                        <div class="realtime-meta">Channel: <b style="color:#d32f2f;">{channel_live_safe}</b></div>
                        <span class="realtime-chip-light">ID: {video_id_live_safe}</span>
                        <span class="realtime-chip">{total_live:,} Komentar</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f'<div class="realtime-kpi"><div class="realtime-kpi-value">{total_live:,}</div><div class="realtime-kpi-label">Total Komentar</div></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown(f'<div class="realtime-kpi"><div class="realtime-kpi-value" style="color:#16a34a">{pos_live:,}</div><div class="realtime-kpi-label">Sentimen Positif</div></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown(f'<div class="realtime-kpi"><div class="realtime-kpi-value" style="color:#FFC107">{neu_live:,}</div><div class="realtime-kpi-label">Sentimen Netral</div></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown(f'<div class="realtime-kpi"><div class="realtime-kpi-value" style="color:#F44336">{neg_live:,}</div><div class="realtime-kpi-label">Sentimen Negatif</div></div>', unsafe_allow_html=True)

        h1, h2 = st.columns([1.2, 1.8])
        with h1:
            st.markdown('<div class="sentiment-header"><div class="sentiment-title">Distribusi Sentimen</div><div class="sentiment-subtitle">Persentase sentimen dari video realtime</div></div>', unsafe_allow_html=True)
        with h2:
            st.markdown('<div class="sentiment-header"><div class="sentiment-title">Statistik Sentimen</div><div class="sentiment-subtitle">Jumlah komentar berdasarkan hasil realtime</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="realtime-sentiment-gap"></div>', unsafe_allow_html=True)

        c_chart1, c_chart2 = st.columns([1.2, 1.8])
        with c_chart1:
            fig = px.pie(
                values=[pos_live, neu_live, neg_live],
                names=["Positif", "Netral", "Negatif"],
                color=["Positif", "Netral", "Negatif"],
                color_discrete_map={"Positif": "#16a34a", "Netral": "#FFC107", "Negatif": "#F44336"}
            )
            fig.update_traces(
                textinfo="percent+label",
                textfont=dict(size=9, color="black", family="Arial Black"),
                domain=dict(x=[0.12, 0.88], y=[0.12, 0.88]),
                hovertemplate="<b>%{label}</b><br>%{value:,} komentar<br>%{percent}"
            )
            fig.update_layout(
                height=210,
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=False,
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_chart2:
            def render_realtime_stat(label, value, total, css_class, icon_path):
                percent = (value / total * 100) if total else 0
                icon_base64 = get_base64_icon(icon_path)
                st.markdown(f"""<div class="stat-item"><div class="stat-label"><img src="data:image/png;base64,{icon_base64}" style="width:16px;"> {label}</div><div class="stat-value">{value:,} komentar ({percent:.1f}%)</div><div class="stat-bar"><div class="stat-fill {css_class}" style="width:{percent}%"></div></div></div>""", unsafe_allow_html=True)
            render_realtime_stat("Positif", pos_live, total_live, "fill-pos", "smile.png")
            render_realtime_stat("Netral", neu_live, total_live, "fill-neu", "neutral.png")
            render_realtime_stat("Negatif", neg_live, total_live, "fill-neg", "sad.png")

        dominant_live = max({"Positif": pos_live, "Netral": neu_live, "Negatif": neg_live}, key={"Positif": pos_live, "Netral": neu_live, "Negatif": neg_live}.get)
        dominant_percent_live = ({"Positif": pos_live, "Netral": neu_live, "Negatif": neg_live}[dominant_live] / total_live * 100) if total_live else 0
        st.markdown(f"""
        <div class="realtime-insight">
            Berdasarkan analisis realtime, video <b>{title_live}</b> memiliki sentimen dominan <b>{dominant_live}</b> sebesar <b>{dominant_percent_live:.1f}%</b>.
            Model evaluasi yang digunakan adalah <b>{metric_live.get('model_used', 'SVM')}</b> dengan F1-Score <b>{metric_live.get('f1-score', 0):.2f}</b>.
        </div>
        """, unsafe_allow_html=True)

        render_sentiment_comment_dropdown(
            df_live,
            key_prefix=f"realtime_{video_id_live_safe}",
            title="Contoh Komentar Berdasarkan Sentimen"
        )

        st.markdown("""<div class="visual-container"><div class="visual-title">Kata Dominan</div><div class="visual-subtitle">WordCloud dan frekuensi kata berdasarkan hasil komentar realtime</div></div>""", unsafe_allow_html=True)
        realtime_words = " ".join(df_live["stemming"].dropna().astype(str))
        if realtime_words.strip():
            vectorizer_live = CountVectorizer()
            X_live = vectorizer_live.fit_transform([realtime_words])
            df_word_live = pd.DataFrame({
                "Kata": vectorizer_live.get_feature_names_out(),
                "Frekuensi": X_live.toarray().flatten()
            }).sort_values("Frekuensi", ascending=False).head(15)
            wc_col, freq_col = st.columns([1.4, 1.8])
            with wc_col:
                if WordCloud is not None:
                    wordcloud_live = WordCloud(
                        width=900,
                        height=450,
                        background_color="white",
                        colormap="Greens",
                        collocations=False
                    ).generate(realtime_words)
                    fig_wc_live, ax_wc_live = plt.subplots(figsize=(8, 4))
                    ax_wc_live.imshow(wordcloud_live, interpolation="bilinear")
                    ax_wc_live.axis("off")
                    wc_buffer = io.BytesIO()
                    fig_wc_live.savefig(wc_buffer, format="png", bbox_inches="tight", pad_inches=0.05, dpi=150)
                    plt.close(fig_wc_live)
                    encoded_wc_live = base64.b64encode(wc_buffer.getvalue()).decode("utf-8")
                    st.markdown(f"""
                    <div class="visual-box">
                        <img src="data:image/png;base64,{encoded_wc_live}" class="realtime-wordcloud-img">
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Library wordcloud belum tersedia, sehingga WordCloud realtime belum dapat ditampilkan.")
            with freq_col:
                fig_words_live = px.bar(df_word_live, x="Kata", y="Frekuensi", color="Frekuensi", color_continuous_scale="Greens")
                fig_words_live.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
                st.plotly_chart(fig_words_live, use_container_width=True)

            realtime_topic_filter = (
                df_live["topic"].dropna().astype(str).str.strip().unique().tolist()
                if "topic" in df_live.columns else None
            )
            pos_words_live = ", ".join(get_sentiment_top_keywords(df_live, "Positif", topic_filter=realtime_topic_filter))
            neu_words_live = ", ".join(get_sentiment_top_keywords(df_live, "Netral", topic_filter=realtime_topic_filter))
            neg_words_live = ", ".join(get_sentiment_top_keywords(df_live, "Negatif", topic_filter=realtime_topic_filter))
            st.markdown(f"""
            <div class="realtime-insight">
                Pada komentar realtime video <b>{title_live_safe}</b>, kata dominan pada sentimen <b>positif</b> adalah
                <b>{pos_words_live}</b>, sentimen <b>netral</b> didominasi oleh <b>{neu_words_live}</b>,
                dan sentimen <b>negatif</b> oleh <b>{neg_words_live}</b>. Pola keyword ini membantu membaca fokus respon audiens pada video tersebut.
            </div>
            """, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.markdown(f'<a href="https://www.youtube.com/watch?v={video_id_live}" target="_blank" class="realtime-watch">Lihat Video Sekarang</a>', unsafe_allow_html=True)
        with btn_col2:
            if st.button("Kembali", key="btn_realtime_back", use_container_width=True):
                st.session_state.realtime_result = None
                st.session_state.realtime_metric = None
                st.session_state.realtime_video_id = ""
                st.rerun()
    elif not realtime_input_error:
        st.markdown("""
        <div class="realtime-empty-state">
            <div class="realtime-empty-icon">🔍</div>
            <div class="realtime-empty-copy">
                <div class="realtime-empty-title">Mulai Analisis Sentiment Secara Realtime</div>
                <div class="realtime-empty-subtitle">Gunakan proses di atas untuk menampilkan ringkasan hasil analisis sentimen berdasarkan video yang telah diproses secara realtime.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# =================== MODEL EVALUATION ====================
# =========================================================
elif menu == "Model Evaluation":

    # ================= HEADER =================
    st.markdown("""
    <div class="page-title">Model Evaluation</div>
    <div class="page-subtitle">
    Evaluasi performa model klasifikasi (SVM, Naive Bayes, LSTM)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <span class="model-eval-filter-anchor"></span>
    <div class="visual-container model-eval-panel-title">
        <div class="visual-title">Perbandingan Metrik Klasifikasi</div>
        <div class="visual-subtitle">Performa model dari Keseluruhan Data</div>
    </div>
    """, unsafe_allow_html=True)

    df_svm_all = load_table_from_postgres("result_svm_balancing")
    df_nb_all = load_table_from_postgres("result_nb_balancing")
    df_lstm_all = load_table_from_postgres("result_lstm_balancing")

    df_svm_all["Model"] = "SVM"
    df_nb_all["Model"] = "Naive Bayes"
    df_lstm_all["Model"] = "LSTM"
    
    df_all = pd.concat([df_svm_all, df_nb_all, df_lstm_all], ignore_index=True)
    df_matrix = df_all[["Model", "Training Data Percentage", "Accuracy", "Precision", "Recall", "F1 Score"]]

    col_filter1, col_filter2 = st.columns([1, 1])

    model_options = df_matrix["Model"].unique()
    train_options = sorted(df_matrix["Training Data Percentage"].unique())

    with col_filter1:
        st.markdown('<div class="model-eval-filter-label">Pilih Model</div>', unsafe_allow_html=True)
        selected_model = st.multiselect(
            label="Pilih Model", 
            options=model_options,
            default=[],
            key="model_eval_global",
            label_visibility="collapsed",
            placeholder="Silakan pilih model"
        )

    with col_filter2:
        st.markdown('<div class="model-eval-filter-label">Training Data (%)</div>', unsafe_allow_html=True)
        selected_train = st.multiselect(
            label="Training Data (%)",
            options=train_options,
            default=[],
            key="train_eval_global",
            label_visibility="collapsed",
            placeholder="Silakan pilih training data (%)"
        )

    if len(selected_model) == 0 or len(selected_train) == 0:
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; border: 1px dashed rgba(255, 255, 255, 0.3);
                    text-align: center; color: white; font-size: 13px; margin-top:-20px;
        ">
            Silakan pilih <b>Model</b> dan <b>Training Data (%)</b> terlebih dahulu untuk melihat hasil analisis.
        </div>
        """, unsafe_allow_html=True)
        
        st.stop()   

    df_filtered = df_matrix[
        (df_matrix["Model"].isin(selected_model)) &
        (df_matrix["Training Data Percentage"].isin(selected_train))
    ]

    st.markdown('<span class="model-eval-table-anchor"></span>', unsafe_allow_html=True)
    st.dataframe(
        df_filtered.style.format({
            "Training Data Percentage": "{:.0f}%",
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}"
        }),
        use_container_width=True
    )

    if not df_filtered.empty:
        best_model = df_filtered.loc[df_filtered["F1 Score"].idxmax()]

        st.markdown(f"""
        <div style="backdrop-filter: blur(10px); padding: 10px; border-radius: 12px; border-left: 6px solid #16a34a;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05); font-size: 12px; line-height: 1.6; color: #ffffff; margin-top: -13px; margin-bottom: 23px;
        ">
            Model terbaik saat ini adalah <b>{best_model['Model']}</b> 
            dengan F1-Score <b>{best_model['F1 Score']:.3f}</b> 
            pada training data <b>{best_model['Training Data Percentage']:.0f}%</b>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Tidak ada data sesuai filter")

    # ================= FILTER & CARD DETAIL PER VIDEO =================
    df_best_per_video = load_table_from_postgres("hasil_best_model_video")
    if not df_best_per_video.empty:
        
        st.markdown("""
        <div class="visual-container model-eval-panel-title model-eval-detail-title">
            <div class="visual-title">Detail Performa Model Per Video</div>
            <div class="visual-subtitle">Detail model terbaik dan metrik performanya dari Setiap Video</div>
        </div>
        """, unsafe_allow_html=True)

        video_titles = sorted(df_best_per_video['title'].unique())
        selected_vids = st.multiselect(placeholder="Cari Judul Video", label="Cari Judul Video", options=video_titles, key="select_video_detail", default=[], label_visibility="collapsed")

        if selected_vids:
            for v_title in selected_vids:
                v_data = df_best_per_video[df_best_per_video['title'] == v_title].iloc[0]
                v_id_eval = str(v_data['video_id'])
                if 'channel' in df_best_per_video.columns:
                    v_channel_eval = v_data['channel']
                elif 'author' in df_best_per_video.columns:
                    v_channel_eval = v_data['author']
                else:
                    video_channel_data = df_sentiment[df_sentiment['video_id'] == v_id_eval]
                    if not video_channel_data.empty and 'channel' in video_channel_data.columns:
                        v_channel_eval = video_channel_data['channel'].iloc[0]
                    elif not video_channel_data.empty and 'author' in video_channel_data.columns:
                        v_channel_eval = video_channel_data['author'].iloc[0]
                    else:
                        v_channel_eval = "Unknown Channel"
                video_comment_data = df_sentiment_video[df_sentiment_video['video_id'] == v_id_eval]
                if not video_comment_data.empty:
                    v_total_comments_eval = int(
                        video_comment_data['Positif'].fillna(0).sum()
                        + video_comment_data['Netral'].fillna(0).sum()
                        + video_comment_data['Negatif'].fillna(0).sum()
                    )
                else:
                    v_total_comments_eval = 0
                
                st.markdown(f"""
                <div class="model-eval-video-card" style="background: white; border-radius: 12px; padding: 14px 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-left: 6px solid #064b22; margin-bottom: 20px;">
                    <div style="color: #064b22; font-size: 15px; font-weight: bold; margin-bottom: 2px;">{v_data['title']}</div>
                    <div style="font-size: 11px; color: #d32f2f; font-weight: bold; margin-bottom: 6px;"><span style="color: #777; font-weight: normal;">Channel:</span> {v_channel_eval}</div>
                    <div style="margin-bottom: 8px;">
                        <span style="font-size: 11px; color: #777; background: #f1f1f1; padding: 4px 10px; border-radius: 6px; display: inline-block; margin-right: 6px;">ID: {v_data['video_id']}</span>
                        <span style="font-size: 11px; color: #ffffff; background: #064b22; padding: 4px 10px; border-radius: 6px; display: inline-block; font-weight: 800;">{v_total_comments_eval:,} Komentar</span>
                    </div>
                    <div style="background: #f8fdf9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px 10px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 13px; color: #333;">Model Terbaik:</span>
                        <span style="background: #064b22; color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 12px;">{v_data['model_used']}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                        <div><div style="font-size: 16px; font-weight: 900; color: #064b22;">{v_data['accuracy']:.3f}</div><div style="font-size: 9px; color: #666; font-weight: bold;">ACCURACY</div></div>
                        <div><div style="font-size: 16px; font-weight: 900; color: #064b22;">{v_data['precision']:.3f}</div><div style="font-size: 9px; color: #666; font-weight: bold;">PRECISION</div></div>
                        <div><div style="font-size: 16px; font-weight: 900; color: #064b22;">{v_data['recall']:.3f}</div><div style="font-size: 9px; color: #666; font-weight: bold;">RECALL</div></div>
                        <div><div style="font-size: 16px; font-weight: 900; color: #064b22;">{v_data['f1-score']:.3f}</div><div style="font-size: 9px; color: #666; font-weight: bold;">F1-SCORE</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
