import plotly.express as px  # <-- EKSİK OLAN BU SATIRDI!
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """Doğruluk, Hassasiyet vb. hesaplar."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0)
    }

def plot_confusion_matrix_custom(y_true, y_pred, model_name):
    """Modelin nerede hata yaptığını gösteren matris."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Heatmap çizimi (px burada kullanılıyor)
    fig = px.imshow(cm, text_auto=True, 
                    labels=dict(x="Tahmin Edilen", y="Gerçek Durum", color="Sayı"),
                    x=['Sağlıklı', 'Glokom'], y=['Sağlıklı', 'Glokom'],
                    title=f"{model_name} Confusion Matrix",
                    color_continuous_scale="Blues")
    return fig

def plot_roc_curves(models_dict, X_test, y_test):
    """Tüm modellerin ROC eğrilerini tek grafikte çizer."""
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            # Olasılık değerlerini al
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            auc_score = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc_score:.2f})", mode='lines'))

    fig.update_layout(
        title="ROC Eğrileri Karşılaştırması",
        xaxis_title="False Positive Rate (Yanlış Alarm)",
        yaxis_title="True Positive Rate (Yakalanan Hasta)",
        legend_title="Modeller"
    )
    return fig