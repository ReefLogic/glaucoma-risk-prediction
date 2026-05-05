import streamlit as st

def render_sidebar():
    """Sol menüdeki ayarları çizer ve kullanıcı seçimlerini döndürür."""
    # st.sidebar.image(...) # İsteğe bağlı logo
    st.sidebar.title("🛠️ Kontrol Paneli")
    
    st.sidebar.header("1. Veri Stratejisi")
    
    # --- SİMÜLASYON KUTUSU ---
    # Bu kutu seçilirse main_app.py veriyi kasıtlı olarak bozacak.
    simulate_missing = st.sidebar.checkbox(
        "🧪 Eksik Veri Simülasyonu", 
        value=False,
        help="Temizlik özelliğini göstermek için veriye yapay boşluklar ekler."
    )
    
    missing_strategy = st.sidebar.selectbox("Eksik Veri Doldurma", ["mean", "median", "drop"])
    outlier_threshold = st.sidebar.slider("Aykırı Değer Eşiği (IQR)", 1.0, 3.0, 1.5)
    
    st.sidebar.divider()
    
    st.sidebar.header("2. Algoritma Seçimi")
    encoding_method = st.sidebar.radio("Kategorik Kodlama", ["label", "onehot"])
    scaling_method = st.sidebar.selectbox("Ölçeklendirme", ["standard", "minmax"])
    
    # --- GÜNCELLENEN MODEL LİSTESİ ---
    # CatBoost çıkarıldı (kurulum hatası nedeniyle), yerine Sklearn'ün SOTA modeli eklendi.
    available_models = [
        "Lojistik Regresyon", 
        "Random Forest", 
        "XGBoost", 
        "LightGBM (Hızlı & Modern)",            # <-- YENİ
        "HistGradientBoosting (Sklearn - SOTA)", # <-- YENİ (CatBoost Alternatifi)
        "SVM", 
        "Yapay Sinir Ağı (MLP)"
    ]
    
    selected_models = st.sidebar.multiselect(
        "Eğitilecek Modeller", 
        available_models,
        # Varsayılan olarak bir Klasik, bir Modern güçlü model seçili gelsin
        default=["Random Forest", "HistGradientBoosting (Sklearn - SOTA)","Lojistik Regresyon","XGBoost","LightGBM (Hızlı & Modern)","SVM","Yapay Sinir Ağı (MLP)"] 
    )
    
    # Tüm ayarları paketleyip döndür
    return {
        "simulate_missing": simulate_missing, 
        "missing_strategy": missing_strategy,
        "outlier_threshold": outlier_threshold,
        "encoding_method": encoding_method,
        "scaling_method": scaling_method,
        "selected_models": selected_models
    }