import streamlit as st
from PIL import Image
import os
from vision_app.model_utils import load_trained_model, predict_single_image

def run_vision_interface():
    st.title("📸 Görüntü Tabanlı Glokom Analizi")
    st.markdown("Derin Öğrenme (CNN) modeli kullanarak retina fotoğraflarını analiz edin.")
    
    # Model Yükleme (Cache ile hızlandırılır)
    @st.cache_resource
    def get_model():
        return load_trained_model()

    model = get_model()
    
    if model is None:
        st.error("⚠️ Eğitilmiş model bulunamadı!")
        st.info("Lütfen önce `python vision_app/train.py` komutunu çalıştırarak modeli eğitin.")
        return

    # Arayüz
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Retina Fotoğrafı Yükle (.jpg, .png)", type=["jpg", "png", "jpeg"])
        
    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Yüklenen Göz Görüntüsü', use_container_width=True)
            
            if st.button("🧠 Analiz Et", type="primary"):
                with st.spinner('Yapay zeka görüntüyü inceliyor...'):
                    result = predict_single_image(model, image)
                    
                    prob_g = result['glaucoma_prob']
                    prob_n = result['normal_prob']
                    
                    st.divider()
                    
                    # Sonuç Gösterimi
                    if result['predicted_class'] == "Glokom":
                        st.error(f"🚨 TESPİT: GLOKOM RİSKİ YÜKSEK (%{prob_g*100:.1f})")
                    else:
                        st.success(f"✅ SONUÇ: NORMAL (%{prob_n*100:.1f})")
                        
                    # İlerleme Çubukları
                    st.write("Detaylı Olasılıklar:")
                    st.write("Glokom:")
                    st.progress(int(prob_g * 100))
                    st.write("Normal:")
                    st.progress(int(prob_n * 100))