import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(df):
    """Korelasyon Matrisi: Hangi sütunlar birbiriyle ilişkili?"""
    # Sadece sayısal sütunlar
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    # text_auto=".2f" ekledik, sayılar daha okunaklı olsun
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", 
                    title="Korelasyon Matrisi", color_continuous_scale='RdBu_r')
    return fig

def plot_target_distribution(df, target_col='Diagnosis'):
    """
    Sınıf Dengesizliği Kontrolü: Kaç hasta, kaç sağlıklı var?
    (GÜNCELLENDİ: Hem metin hem sayı formatını destekler)
    """
    if target_col not in df.columns:
        return px.bar(title="Hata: Hedef sütun bulunamadı!")

    counts = df[target_col].value_counts().reset_index()
    counts.columns = ['Durum', 'Kişi Sayısı']
    
    # AKILLI HARİTALAMA: 
    # Veri setinde "1/0" da olsa, "Glaucoma/No Glaucoma" da olsa Türkçe göster.
    mapping = {
        1: 'Glokom (Hasta)', 
        0: 'Sağlıklı',
        'Glaucoma': 'Glokom (Hasta)',
        'No Glaucoma': 'Sağlıklı'
    }
    
    # Eşleşenleri değiştir, eşleşmeyenleri (listede yoksa) olduğu gibi bırak (fillna)
    counts['Durum'] = counts['Durum'].map(mapping).fillna(counts['Durum'])
    
    fig = px.bar(counts, x='Durum', y='Kişi Sayısı', color='Durum', 
                 title="Hedef Değişken Dağılımı", text_auto=True)
    return fig

def plot_interactive_scatter(df, x_col, y_col, color_col='Diagnosis'):
    """Kullanıcı seçimine göre saçılım grafiği."""
    df_plot = df.copy()
    
    # Grafikteki renk lejandını da Türkçeleştirelim
    if color_col in df_plot.columns:
         mapping = {1: 'Glokom', 0: 'Sağlıklı', 'Glaucoma': 'Glokom', 'No Glaucoma': 'Sağlıklı'}
         # map sonrası string yapıyoruz ki Plotly hata vermesin
         df_plot[color_col] = df_plot[color_col].map(mapping).fillna(df_plot[color_col]).astype(str)

    fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_col,
                     title=f"{x_col} ve {y_col} İlişkisi", opacity=0.7)
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Özellik Önemi: Model karar verirken en çok neye baktı?"""
    importances = None
    
    # 1. Durum: Ağaç Tabanlı Modeller (Random Forest, XGBoost)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
    # 2. Durum: Lineer Modeller (Lojistik Regresyon) - Bunu ekledik ki hata vermesin
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
        
    if importances is not None:
        df_imp = pd.DataFrame({'Özellik': feature_names, 'Önem': importances})
        # Grafikte düzgün durması için sıralayalım
        df_imp = df_imp.sort_values(by='Önem', ascending=True)
        
        fig = px.bar(df_imp, x='Önem', y='Özellik', orientation='h', 
                     title=f"{model_name} İçin Özellik Önemi")
        return fig
    else:
        return None