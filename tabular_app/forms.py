import streamlit as st

def patient_entry_form():
    """
    Kullanıcıdan Türkçe veri alır, ancak sisteme
    CSV dosyasındaki orijinal formatta (İngilizce terimlerle) veri döndürür.
    """
    st.markdown("#### 📋 Yeni Hasta Veri Girişi")
    st.info("Lütfen aşağıdaki değerleri hastanın tahlil sonuçlarına göre giriniz.")

    with st.form("new_patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # YAŞ
            age = st.number_input("Yaş", min_value=18, max_value=120, value=50)
            
            # CİNSİYET
            gender_tr = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
            gender_val = "Male" if gender_tr == "Erkek" else "Female"
            
            # GÖZ TANSİYONU (IOP)
            iop = st.number_input("Göz İçi Basıncı (IOP)", min_value=5.0, max_value=60.0, value=15.0, 
                                  help="Normal değer: 10-21 mmHg")
            
        with col2:
            # KORNEA KALINLIĞI (Pachymetry)
            pachy = st.number_input("Kornea Kalınlığı (Pachymetry)", min_value=300.0, max_value=800.0, value=550.0, 
                                    help="Ortalama: 540 µm")
            
            # SİNİR HASARI (CDR)
            cdr = st.number_input("Optik Çukur/Disk Oranı (CDR)", min_value=0.0, max_value=1.0, value=0.4, step=0.01,
                                  help="0.5 üzeri değerler risklidir.")
            
            # AİLE GEÇMİŞİ
            history_tr = st.selectbox("Ailede Glokom Var mı?", ["Evet", "Hayır"])
            history_val = "Yes" if history_tr == "Evet" else "No"
            
        # Gönderme Butonu
        submitted = st.form_submit_button("🔍 Analiz Et")
        
        if submitted:
            # BURASI ÇOK ÖNEMLİ:
            # Anahtarlar (Keys) tam olarak CSV dosyasındaki sütun adları olmalı!
            data = {
                'Age': age,
                'Gender': gender_val,                         # Male/Female
                'Intraocular Pressure (IOP)': iop,
                'Cup-to-Disc Ratio (CDR)': cdr,
                'Pachymetry': pachy,
                'Family History': history_val                 # Yes/No
            }
            return True, data
        
        return False, None