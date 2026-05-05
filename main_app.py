import streamlit as st

# Sayfa ayarını EN BAŞTA yapmalıyız
st.set_page_config(
    page_title="Glokom Hibrit Teşhis Sistemi", 
    layout="wide", 
    page_icon="👁️"
)

# Kendi modüllerimizi çağırıyoruz
from tabular_app.app import run_tabular_interface
from vision_app.interface import run_vision_interface
# --- YENİ EKLENEN RAPOR MODÜLÜ ---
from project_summary import run_project_report 

# --- CSS İLE GÖRSEL DÜZENLEME (GÜNCELLENDİ) ---
# Not: Sol menünün (Sidebar) okunmama sorununu çözmek için bu kısmı
# geçici olarak yorum satırına aldım. Streamlit'in kendi temasını kullanmak
# şu an en temiz görüntüyü verecektir.
# st.markdown("""
#     <style>
#     .main { background-color: #f5f5f5; }
#     h1 { color: #2c3e50; }
#     .stSidebar { background-color: #dfe6e9; }
#     </style>
# """, unsafe_allow_html=True)

# --- KENAR ÇUBUĞU (NAVIGASYON) ---
# Logo
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
st.sidebar.title("Navigasyon")

# --- MENÜYE 3. SEÇENEĞİ EKLEDİK ---
app_mode = st.sidebar.radio(
    "Modül Seçiniz:",
    [
        "📊 Sayısal Veri Analizi", 
        "📸 Görüntü ile Teşhis", 
        "📄 Proje Raporu ve Sonuçlar"
    ]
)

st.sidebar.info("Bu sistem Çok Modlu (Multimodal) Yapay Zeka mimarisi kullanmaktadır.")

# --- SAYFA YÖNLENDİRME ---
if app_mode == "📊 Sayısal Veri Analizi":
    # Eski sistem (Tabular)
    run_tabular_interface()

elif app_mode == "📸 Görüntü ile Teşhis":
    # Yeni sistem (Görüntü İşleme)
    run_vision_interface()

elif app_mode == "📄 Proje Raporu ve Sonuçlar":
    # --- YENİ RAPOR EKRANI ---
    run_project_report()