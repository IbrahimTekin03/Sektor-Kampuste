import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import os
from src.config import IMG_SIZE, CLASS_TRANSLATIONS

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="Photo Scan AI",
    page_icon="",
    layout="wide"
)

# --- 2. ÖZEL CSS (TASARIMIN KALBİ BURASI) ---
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(to bottom right, #e8f5e9, #ffffff);
    }

    /* Başlık Stili */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1b5e20;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-title {
        text-align: center;
        color: #4caf50;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }

    /* Kart Yapısı (Beyaz Kutular) */
    .css-card {
        background-color: white;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Buton Tasarımı */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #43a047, #66bb6a);
        color: white;
        border: none;
        padding: 18px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(67, 160, 71, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(67, 160, 71, 0.4);
    }

    /* Yükleme Alanı */
    .stFileUploader {
        text-align: center;
    }

    /* Metrik Kutuları */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 3. MODEL YÜKLEME ---
@st.cache_resource
def load_ai_assets():
    model_path = os.path.join("models", "animals10_gpu_model.h5")
    class_path = os.path.join("models", "class_names.pkl")

    if not os.path.exists(model_path) or not os.path.exists(class_path):
        return None, None

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(class_path, 'rb') as f:
        raw_classes = pickle.load(f)
    return model, raw_classes


model, raw_classes = load_ai_assets()

# --- 4. BAŞLIK VE GİRİŞ ---
st.markdown('<div class="main-title">Photo Scan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Yapay Zeka Destekli Doğa Gözlem ve Analiz Sistemi</div>', unsafe_allow_html=True)

# --- 5. ANA DÜZEN (İKİ KOLON) ---
# Container kullanarak içeriği ortalıyoruz
with st.container():
    col1, col2 = st.columns([1, 1], gap="large")

    # SOL KOLON: RESİM YÜKLEME VE GÖSTERİM
    with col1:
        st.markdown("### Görüntü Yükleme")
        st.write("Analiz etmek istediğiniz fotoğrafı aşağıya sürükleyin.")

        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Resme hafif gölge ve yuvarlak köşe ekleyelim (Streamlit içinde stil)
            st.image(image, caption='Seçilen Görüntü', use_container_width=True)


    # SAĞ KOLON: BUTON VE SONUÇLAR
    with col2:
        st.markdown("### Analiz Paneli")

        if uploaded_file is not None:
            st.write("Görüntü hazır. Yapay zeka motorunu başlatmak için butona basın.")

            # Dev Buton
            if st.button('Taramayı Başlat ✨'):
                if model is None:
                    st.error("⚠️ Model bulunamadı! Lütfen eğitimi tamamlayın.")
                else:
                    with st.spinner('Pikseller işleniyor...'):
                        # Görüntü İşleme
                        img = image.resize(IMG_SIZE)
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        # Tahmin
                        predictions = model.predict(img_array)

                        # Verileri Düzenle
                        predicted_idx = np.argmax(predictions)
                        predicted_raw_name = raw_classes[predicted_idx]
                        confidence = 100 * np.max(predictions)
                        turkish_name = CLASS_TRANSLATIONS.get(predicted_raw_name, predicted_raw_name)

                        # Sabit Eşik Değeri (Artık kodun içinde gizli)
                        CONFIDENCE_THRESHOLD = 60

                        st.write("---")  # Ayırıcı çizgi

                        # SONUÇ MANTIĞI
                        if confidence < CONFIDENCE_THRESHOLD:
                            st.warning(f"⚠️ **Tanımlanamayan Nesne**")
                            st.markdown(f"""
                            Sistem bu görüntüden tam emin olamadı.
                            **Tahmin:** {turkish_name} (%{confidence:.1f})
                            *Lütfen daha net bir hayvan fotoğrafı deneyin.*
                            """)
                        else:
                            # Başarılı Sonuç
                            st.success("✅ Analiz Başarıyla Tamamlandı")

                            # Büyük İstatistik Gösterimi
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.metric("Tespit Edilen Tür", turkish_name.upper())
                            with m_col2:
                                st.metric("Güven Skoru", f"%{confidence:.1f}")

                            # Grafik Alanı
                            st.markdown("#### Olasılık Dağılımı")

                            # Pandas ile Grafik Verisi
                            probs = predictions[0]
                            class_data = {
                                "Hayvan": [CLASS_TRANSLATIONS.get(cls, cls) for cls in raw_classes],
                                "Olasılık": probs
                            }
                            df = pd.DataFrame(class_data)
                            # En yüksek 5 tahmini al ve sırala
                            df = df.sort_values(by="Olasılık", ascending=False).head(5)

                            # Renkli Bar Chart
                            st.bar_chart(df, x="Hayvan", y="Olasılık", color="#66bb6a")

        else:
            # Resim yokken sağ taraf boş kalmasın diye bilgilendirme
            st.markdown("""
            <div style="background-color: #f1f8e9; padding: 20px; border-radius: 10px; color: #558b2f;">
                <h4>Nasıl Çalışır?</h4>
                <ol>
                    <li>Sol taraftan bir hayvan fotoğrafı yükleyin.</li>
                    <li>Sistem otomatik olarak görüntüyü işler.</li>
                    <li>Sonuçlar ve grafikler bu alanda belirir.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)