import streamlit as st
import pandas as pd

def run_project_report():
    st.title("📄 Proje Tartışma ve Sonuç Raporu")
    st.markdown("---")

    # 1. YÖNETİCİ ÖZETİ
    st.header("1. Yönetici Özeti ve Hibrit Mimari")
    st.info("""
    **Glokom-Hibrit Projesi**, literatürdeki tek yönlü çalışmaların aksine **"Multimodal (Çok Modlu)"** bir yaklaşım benimser. Sistemimiz, hastayı tek bir veri tipine göre değil, bütüncül olarak değerlendirir.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.success("**Modül A: Sayısal Analiz**")
        st.write("- **Veri:** Klinik Ölçümler (CSV)")
        st.write("- **Model:** Random Forest / XGBoost")
        st.write("- **Odak:** Göz Tansiyonu, Yaş, Kornea Kalınlığı")
    with col2:
        st.warning("**Modül B: Görsel Analiz**")
        st.write("- **Veri:** Retina Fotoğrafları (Fundus)")
        st.write("- **Model:** Derin Öğrenme (CNN - ResNet18)")
        st.write("- **Odak:** Optik Disk Çukurlaşması (Cupping)")

    st.divider()

    # 2. KARŞILAŞTIRMALI ANALİZ (TABLO)
    st.header("2. Literatür ve Mevcut Durum Analizi")
    st.write("Geliştirdiğimiz sistemin, literatürdeki diğer yöntemlerle karşılaştırması aşağıdadır:")

    comparison_data = {
        "Özellik": ["Veri Tipi", "Kullanılan Teknoloji", "Teşhis Güvenilirliği", "Hata Payı"],
        "Sadece Sayısal Yöntemler": ["Sadece CSV (Sayılar)", "Makine Öğrenmesi (ML)", "Orta (Görüntü Yok)", "Yüksek (Yanlış Pozitif)"],
        "Sadece Görüntü İşleme": ["Sadece Piksel (Resim)", "Derin Öğrenme (CNN)", "Yüksek (Klinik Veri Yok)", "Orta (Görüntü Kalitesine Bağlı)"],
        "✅ Bizim Hibrit Sistem": ["CSV + Resim (Multimodal)", "Ensemble (RF + ResNet18)", "Maksimum (Çapraz Doğrulama)", "Minimize Edilmiş"]
    }
    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)

    st.divider()

    # 3. BULGULAR VE TARTIŞMA
    st.header("3. Bulgular ve Tartışma")
    
    with st.expander("📌 Bulgu 1: ResNet18'in Üstünlüğü (Görsel Modül)", expanded=True):
        st.markdown("""
        Kaggle (SSHikamaru/Glaucoma) veri seti üzerinde yapılan eğitimlerde, **ResNet18** mimarisinin 
        daha sığ ağlara göre optik diskteki mikro deformasyonları yakalamada daha başarılı olduğu görülmüştür. 
        Transfer Learning (ImageNet ağırlıkları) kullanımı, eğitim süresini kısaltmış ve **Loss değerini 0.2 seviyelerine** düşürmüştür.
        """)
        
    with st.expander("📌 Bulgu 2: Klinik Doğrulama (Sayısal Modül)", expanded=True):
        st.markdown("""
        Görüntü işleme modelleri bazen ışık parlamalarını hastalık sanabilir. Ancak geliştirdiğimiz sistemde, 
        sayısal modül hastanın göz tansiyonunu (IOP) normal bulursa, sistem doktora **"Görüntü şüpheli ama klinik veriler temiz"** uyarısı vererek gereksiz paniği önlemektedir.
        """)

    st.divider()

    # 4. REFERANSLAR
    st.header("4. Kaynakça ve Veri Setleri")
    st.markdown("""
    Bu çalışmada kullanılan yöntemler ve veri setleri aşağıdaki akademik kaynaklara dayanmaktadır:
    
    * **Veri Setleri:** [Kaggle Glaucoma Detection Dataset (SSHikamaru)](https://www.kaggle.com/datasets/sshikamaru/glaucoma-detection) [Kaggle Glaucoma Detection Dataset(INCRIBO)] (https://www.kaggle.com/datasets/teamincribo/glaucoma-detection-dataset)
    * **Referans Makale 1:** Chen, X., et al. (2015). *"Glaucoma detection based on deep convolutional neural networks."*
    * **Referans Makale 2:** He, K., et al. (2016). *"Deep Residual Learning for Image Recognition (ResNet)."*
    * **Referans Makale 3:** Al-Gharaibeh, B., et al. (2021). *"Hybrid deep learning approach for glaucoma detection."*
    """)
    
    st.caption("© 2026 Glokom AI Lab - Bitirme Projesi Raporu")