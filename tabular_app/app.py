import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- IMPORT YOLLARI GÜNCELLENDİ (Artık hepsi yan yana) ---
from tabular_app.loader import load_data
from tabular_app.preprocessor import DataPreprocessor
from tabular_app.charts import plot_correlation_heatmap, plot_target_distribution, plot_interactive_scatter
from tabular_app.trainers import ModelTrainer
from tabular_app.metrics import calculate_metrics, plot_confusion_matrix_custom, plot_roc_curves
from tabular_app.sidebar import render_sidebar
from tabular_app.forms import patient_entry_form

def run_tabular_interface():
    """
    Eski main_app.py kodlarının tamamı bu fonksiyonun içine alındı.
    Böylece ana main.py dosyasından çağrılabilir hale geldi.
    """
    
    # --- HAFIZA AYARLARI ---
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = {} 
    if 'input_order' not in st.session_state:
        st.session_state['input_order'] = [] 
    if 'scaler' not in st.session_state:
        st.session_state['scaler'] = None 

    # 1. UI - Sidebar'ı buraya özel çağırıyoruz
    with st.sidebar:
        st.divider()
        st.caption("📊 Sayısal Analiz Ayarları")
        settings = render_sidebar()

    st.header("📊 Klinik Veri Analizi (Sayısal)")
    st.markdown("Hasta tahlil sonuçları (CSV) üzerinden analiz modundasınız.")

    # 2. VERİ YÜKLEME
    # DÜZELTME: Veri yolu güncellendi -> 'data/tabular/...'
    df_loaded = load_data("data/tabular/glokomData.csv")

    if df_loaded is None:
        st.error("Veri seti bulunamadı! Lütfen 'data/tabular/glokomData.csv' dosyasını kontrol edin.")
        return

    # --- KRİTİK: Deep Copy ---
    df_raw = df_loaded.copy(deep=True)

    # Gereksiz sütunları at
    selected_columns = [
        'Age', 'Gender', 'Intraocular Pressure (IOP)', 
        'Cup-to-Disc Ratio (CDR)', 'Pachymetry', 
        'Family History', 'Diagnosis'
    ]
    try:
        df_raw = df_raw[selected_columns]
    except KeyError:
        pass

    # --- SİMÜLASYON MODU ---
    if settings.get('simulate_missing'):
        np.random.seed(42)
        cols_missing = ['Intraocular Pressure (IOP)', 'Pachymetry', 'Cup-to-Disc Ratio (CDR)']
        for col in cols_missing:
            if col in df_raw.columns:
                mask = np.random.rand(len(df_raw)) < 0.10
                df_raw.loc[mask, col] = np.nan
        
        if 'Intraocular Pressure (IOP)' in df_raw.columns:
            outlier_mask = np.random.rand(len(df_raw)) < 0.05
            df_raw.loc[outlier_mask, 'Intraocular Pressure (IOP)'] = np.random.uniform(50, 90, size=outlier_mask.sum())

        if 'Cup-to-Disc Ratio (CDR)' in df_raw.columns:
            outlier_mask_cdr = np.random.rand(len(df_raw)) < 0.05
            df_raw.loc[outlier_mask_cdr, 'Cup-to-Disc Ratio (CDR)'] = np.random.uniform(1.5, 3.0, size=outlier_mask_cdr.sum())

        if 'Intraocular Pressure (IOP)' in df_raw.columns and 'Diagnosis' in df_raw.columns:
             high_iop_mask = df_raw['Intraocular Pressure (IOP)'] > 25
             df_raw.loc[high_iop_mask, 'Diagnosis'] = 'Glaucoma' 

        st.warning("🧪 SİMÜLASYON AKTİF: Veriye yapay boşluklar, aykırı değerler ve GÜÇLÜ KORELASYON eklendi!")

    # --- VERİ ÖZETİ ---
    col_info1, col_info2 = st.columns([1, 3])
    with col_info1:
        st.metric(label="Toplam Hasta Kaydı", value=len(df_raw), delta="Aktif Veri Seti")
    with col_info2:
        with st.expander("📂 Veri Setine Göz At (Önizleme)"):
            st.dataframe(df_raw.head(100), use_container_width=True, height=200)
            st.caption("Not: Performans için ilk 100 kayıt gösterilmektedir.")
    
    st.divider() 

    tab1, tab2, tab3 = st.tabs(["📊 Veri Analizi", "🧠 Model Eğitimi", "⚡ Gerçek Zamanlı Tahmin"])

    # --- VERİ İŞLEME ---
    preprocessor = DataPreprocessor(df_raw)
    df_clean = preprocessor.handle_missing_values(strategy=settings['missing_strategy'])
    
    # Aykırı Değerler
    for col in ['Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - (settings['outlier_threshold'] * IQR)
            upper_limit = Q3 + (settings['outlier_threshold'] * IQR)
            df_clean[col] = np.where(df_clean[col] > upper_limit, upper_limit, df_clean[col])
            df_clean[col] = np.where(df_clean[col] < lower_limit, lower_limit, df_clean[col])
    
    # Encoding
    if 'Gender' in df_clean.columns:
        df_clean['Gender'] = df_clean['Gender'].astype(str).str.strip().map({'Male': 1, 'Female': 0}).fillna(0)
    if 'Family History' in df_clean.columns:
        df_clean['Family History'] = df_clean['Family History'].astype(str).str.strip().map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Scaling
    X_scaled, y, scaler = preprocessor.scale_features(method=settings['scaling_method'])
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- SEKME 1: ANALİZ ---
    with tab1:
        st.info("🕵️‍♂️ Veri Kalite Raporu")
        col_test1, col_test2, col_test3 = st.columns(3)
        col_test1.metric("Ham Verideki Boşluk", f"{df_raw.isnull().sum().sum()} Adet")
        col_test2.metric("İşlem Sonrası Hasta Sayısı", f"{len(df_clean)}", delta=f"{len(df_clean) - len(df_raw)} (Kayıp)")
        col_test3.metric("Temizlik Sonrası Kalan Boşluk", f"{df_clean.isnull().sum().sum()} Adet")
        
        with st.expander("📏 Aykırı Değer Etkisi (Baskılama Kontrolü)", expanded=True):
             ac1, ac2 = st.columns(2)
             if 'Intraocular Pressure (IOP)' in df_raw.columns:
                raw_max_iop = df_raw['Intraocular Pressure (IOP)'].max()
                clean_max_iop = df_clean['Intraocular Pressure (IOP)'].max()
                ac1.metric("Maksimum Göz Tansiyonu (IOP)", f"{clean_max_iop:.2f}", delta=f"{clean_max_iop - raw_max_iop:.2f} (Değişim)", delta_color="inverse") 
             if 'Cup-to-Disc Ratio (CDR)' in df_raw.columns:
                raw_max_cdr = df_raw['Cup-to-Disc Ratio (CDR)'].max()
                clean_max_cdr = df_clean['Cup-to-Disc Ratio (CDR)'].max()
                ac2.metric("Maksimum CDR Oranı", f"{clean_max_cdr:.2f}", delta=f"{clean_max_cdr - raw_max_cdr:.2f} (Değişim)", delta_color="inverse")
        
        st.divider()
        st.subheader("Veri Analizi")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_target_distribution(df_raw), use_container_width=True)
        with c2: 
            df_viz = df_raw.copy()
            # ... (Korelasyon kodları aynen kalsın) ...
            # Özet geçmek için burayı kısaltmıyorum, senin kodun aynen burada çalışacak.
            df_viz_numeric = df_viz.select_dtypes(include=[np.number])
            st.plotly_chart(plot_correlation_heatmap(df_viz_numeric), use_container_width=True)
        
        col_x = st.selectbox("X Ekseni", df_clean.columns, index=0)
        col_y = st.selectbox("Y Ekseni", df_clean.columns, index=2)
        st.plotly_chart(plot_interactive_scatter(df_raw, col_x, col_y), use_container_width=True)

    # --- SEKME 2: MODEL EĞİTİMİ ---
    with tab2:
        st.write(f"Modeller: {settings['selected_models']}")
        if st.button("🚀 Modelleri Eğit"):
            if not settings['selected_models']:
                st.warning("Model seçiniz.")
            else:
                trainer = ModelTrainer()
                st.session_state['trained_models'] = {}
                st.session_state['input_order'] = X_train.columns.tolist()
                st.session_state['scaler'] = scaler 
                results = []
                progress = st.progress(0)
                
                for idx, name in enumerate(settings['selected_models']):
                    model = trainer.get_model(name, {})
                    model.fit(X_train, y_train)
                    st.session_state['trained_models'][name] = model
                    y_pred = model.predict(X_test)
                    metrics = calculate_metrics(y_test, y_pred)
                    metrics['Model'] = name
                    results.append(metrics)
                    progress.progress((idx+1)/len(settings['selected_models']))
                
                st.success("Eğitim Tamamlandı!")
                st.table(pd.DataFrame(results).set_index('Model'))
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(plot_roc_curves(st.session_state['trained_models'], X_test, y_test), use_container_width=True)
                with c2: 
                    first = list(st.session_state['trained_models'].keys())[0]
                    st.plotly_chart(plot_confusion_matrix_custom(y_test, st.session_state['trained_models'][first].predict(X_test), first), use_container_width=True)

    # --- SEKME 3: TAHMİN ---
    with tab3:
        is_submitted, patient_data = patient_entry_form()
        if is_submitted:
            if not st.session_state['trained_models']:
                st.error("⚠️ Önce 'Model Eğitimi' sekmesinden modelleri eğitmelisiniz!")
            else:
                input_df = pd.DataFrame([patient_data])
                try:
                    input_df['Gender'] = input_df['Gender'].astype(str).str.strip().map({'Male': 1, 'Female': 0}).fillna(0)
                    input_df['Family History'] = input_df['Family History'].astype(str).str.strip().map({'Yes': 1, 'No': 0}).fillna(0)
                except Exception as e: st.error(f"Dönüşüm hatası: {e}")
                
                expected_cols = st.session_state['input_order']
                final_input = pd.DataFrame(0, index=[0], columns=expected_cols)
                for col in input_df.columns:
                    if col in final_input.columns: final_input[col] = input_df[col]
                
                if st.session_state['scaler']:
                    try: final_input_scaled = st.session_state['scaler'].transform(final_input)
                    except: final_input_scaled = final_input
                else: final_input_scaled = final_input

                cols = st.columns(len(st.session_state['trained_models']))
                best_prob = 0
                best_model = ""
                for idx, (name, model) in enumerate(st.session_state['trained_models'].items()):
                    if hasattr(model, "predict_proba"): prob = model.predict_proba(final_input_scaled)[0][1]
                    else: prob = model.predict(final_input_scaled)[0]
                    decision = "RİSKLİ" if prob > 0.5 else "NORMAL"
                    if prob > best_prob: best_prob = prob; best_model = name
                    with cols[idx]:
                        st.metric(name, f"%{prob*100:.1f}", delta=decision)
                
                st.divider()
                if best_prob > 0.5: st.error(f"🚨 DİKKAT: {best_model} %{best_prob*100:.1f} ihtimalle GLOKOM riski tespit etti.")
                else: st.success("✅ GÜVENLİ: Sonuçlar normal görünüyor.")