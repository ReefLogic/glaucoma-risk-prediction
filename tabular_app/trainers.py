from sklearn.linear_model import LogisticRegression
# HistGradientBoostingClassifier eklendi (CatBoost yerine modern alternatif)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# LightGBM Kütüphanesi
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

class ModelTrainer:
    def get_model(self, model_name, params):
        """Seçilen isme ve parametrelere göre modeli hazırlar."""
        
        # --- MEVCUT MODELLER (DOKUNULMADI) ---
        if model_name == "Lojistik Regresyon":
            # Pseudocode: Çizgi çek (y = mx + c) ve Sigmoid ile 0-1 arasına sıkıştır.
            return LogisticRegression(C=params.get('C', 1.0), max_iter=1000)
            
        elif model_name == "Random Forest":
            # Pseudocode: Rastgele alt veri setleri oluştur -> Karar Ağaçları kur -> Oylama yap.
            return RandomForestClassifier(n_estimators=params.get('n_estimators', 100), 
                                          max_depth=params.get('max_depth', 10),
                                          random_state=42)
                                          
        elif model_name == "XGBoost":
            # Pseudocode: Hataları hesapla -> Yeni ağaç ekle (hatayı azaltacak şekilde) -> Güncelle.
            return xgb.XGBClassifier(n_estimators=params.get('n_estimators', 100),
                                     learning_rate=params.get('learning_rate', 0.1),
                                     eval_metric='logloss',
                                     random_state=42)
                                     
        elif model_name == "SVM":
            # Pseudocode: En yakın noktalara (destek vektörleri) bak -> Aradaki marjı maksimize et.
            return SVC(C=params.get('C', 1.0), kernel=params.get('kernel', 'rbf'), probability=True, random_state=42)
            
        elif model_name == "Yapay Sinir Ağı (MLP)":
            # Pseudocode: Girdi -> Gizli Katman (Ağırlık * Girdi + Bias) -> Aktivasyon -> Çıktı.
            return MLPClassifier(hidden_layer_sizes=(params.get('hidden_layer', 100),), 
                                 max_iter=1000, random_state=42)

        # --- YENİ EKLENEN MODERN MODELLER ---
        
        elif model_name == "LightGBM (Hızlı & Modern)":
            # Microsoft (2017): Leaf-wise büyüme stratejisi ile çok hızlı ve hafiftir.
            if LGBMClassifier:
                return LGBMClassifier(n_estimators=params.get('n_estimators', 100),
                                      learning_rate=params.get('learning_rate', 0.1),
                                      random_state=42,
                                      verbose=-1) # Gereksiz uyarıları kapat
            else:
                return None

        elif model_name == "HistGradientBoosting (Sklearn - SOTA)":
            # Scikit-Learn (2019+): CatBoost ve LightGBM mimarisini kullanan yerel kütüphane.
            # Kurulum gerektirmez, çok hızlıdır ve "Missing Value"ları otomatik yönetir.
            # Not: Bu modelde n_estimators yerine max_iter parametresi kullanılır.
            return HistGradientBoostingClassifier(max_iter=params.get('n_estimators', 100),
                                                  learning_rate=params.get('learning_rate', 0.1),
                                                  random_state=42)
        
        else:
            return None