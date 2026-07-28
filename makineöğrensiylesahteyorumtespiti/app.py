import streamlit as st
import joblib
import pandas as pd
import scipy.sparse as sp
import numpy as np
import re
import nltk
# 1. Sayfa Ayarları
st.set_page_config(page_title="Sahte Yorum Dedektörü", page_icon="🛡️", layout="centered")

# 2. NLTK Stopwords Yükleme Mekanizması (En Kararlı Hali)
@st.cache_resource
def load_stopwords():
  
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))
try:
    stop_words = load_stopwords()
except Exception as e:
    st.error(f"Stopwords yüklenirken hata oluştu: {e}")

# Canlı metin temizleme fonksiyonu
def clean_text_live(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Noktalamaları temizler
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# 3. Modelleri Yükleme
@st.cache_resource 
def load_assets():
    model = joblib.load('best_svm_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    return model, tfidf, scaler

try:
    model, tfidf, scaler = load_assets()
except Exception as e:
    st.error(f"Model dosyaları yüklenirken hata oluştu! Hata: {e}")

# 4. Arayüz Tasarımı
st.title("🛡️ Amazon Sahte Yorum Tespit Sistemi")
st.markdown("---")

st.subheader("✍️ Analiz Edilecek Yorum Metni")
user_review = st.text_area("Müşteri yorumunu buraya yapıştırın:", height=150, placeholder="Örn: This product is amazing...", key="unique_review_key")

st.subheader("📊 Yoruma Ait Davranışsal Meta Veriler")

col1, col2 = st.columns(2)
with col1:
    val_overall = st.slider("Yıldız Puanı (Overall Rating):", 1.0, 5.0, 4.0, step=1.0)
    val_helpful = st.number_input("Yorumu 'Faydalı' Bulan Kişi Sayısı:", min_value=0, value=0, step=1)
with col2:
    val_total = st.number_input("Yorum için Kullanılan Toplam Oy Sayısı:", min_value=0, value=0, step=1)
    val_day = st.number_input("Yorumun Paylaşılmasının Üzerinden Geçen Gün Sayısı:", min_value=0, value=100, step=1)

# 5. Tahmin ve Karar Dünyası
st.markdown("---")
if st.button("🔍 Yorumu Analiz Et", use_container_width=True):
    cleaned_input_text = user_review.strip()
    
    if cleaned_input_text == "":
        st.warning("Lütfen analiz etmek için geçerli bir yorum metni girin!")
    elif val_total < val_helpful:
        st.error("HATA: Toplam oy sayısı, faydalı bulan kişi sayısından az olamaz!")
    else:
        with st.spinner("Yapay zeka modeli verileri analiz ediyor..."):
            # A - Metni temizle ve TF-IDF'e çevir
            live_cleaned = clean_text_live(cleaned_input_text)
            text_matrix = tfidf.transform([live_cleaned])
            
            # B - Sayısal verileri hazırla ve ölçeklendir
            raw_numerical_data = np.array([[float(val_overall), float(val_helpful), float(val_total), float(val_day)]])
            scaled_numerical_data = scaler.transform(raw_numerical_data)
            
            # C - Matrisleri birleştir
            X_live_input = sp.hstack((text_matrix, scaled_numerical_data), format='csr')
            
            # D - Tahminleri üret (İsim çakışmaları tamamen çözüldü)
            final_pred = model.predict(X_live_input)[0]
            final_proba = model.predict_proba(X_live_input)[0]
            
            st.subheader("🚨 Analiz Sonucu")
            
            # E - Sonuç Ekranı ve İlerleme Çubukları
            if final_pred == 1:
                st.error("**Sistem Kararı: SAHTE / MANİPÜLATİF YORUM**")
                st.progress(float(final_proba[1]))
                st.write(f"Model Güveni: %{final_proba[1]*100:.2f} ihtimalle manipülasyon içeriyor.")
            else:
                st.success("**Sistem Kararı: GERÇEK / ORGANİK YORUM**")
                st.progress(float(final_proba[0]))
                st.write(f"Model Güveni: %{final_proba[0]*100:.2f} ihtimalle gerçek kullanıcı deneyimi.")