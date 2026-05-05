import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        # Hedef değişkeni sayıya çevirelim
        if 'Diagnosis' in self.df.columns:
            self.df['Diagnosis'] = self.df['Diagnosis'].apply(lambda x: 1 if 'No Glaucoma' not in x else 0)
    
    def handle_missing_values(self, strategy='mean'):
        """Eksik verileri seçilen stratejiye göre doldurur."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'drop':
            self.df.dropna(inplace=True)
            
        # Kategorik boşlukları mod ile doldur
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if not self.df[col].mode().empty:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
        return self.df

    def handle_outliers(self, column, threshold=1.5):
        """IQR Yöntemi ile aykırı değerleri baskılar."""
        if column in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            
            self.df[column] = np.where(self.df[column] < lower_bound, lower_bound, self.df[column])
            self.df[column] = np.where(self.df[column] > upper_bound, upper_bound, self.df[column])
        return self.df

    def encode_categorical(self, method='label'):
        """Kategorik verileri sayıya çevirir."""
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        if method == 'label':
            le = LabelEncoder()
            for col in cat_cols:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        elif method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
        return self.df

    def scale_features(self, method='standard', target_col='Diagnosis'):
        """
        Verileri ölçekler ve SCALER nesnesini de döndürür.
        """
        # Sadece sayısal sütunları al (Target hariç)
        X = self.df.drop(columns=[target_col], errors='ignore')
        y = self.df[target_col] if target_col in self.df.columns else None
        
        # Sadece sayısal veriler üzerinde çalış (Garanti olsun)
        X = X.select_dtypes(include=[np.number])
        
        if method == 'standard':
            scaler = StandardScaler()
        else: 
            scaler = MinMaxScaler()
            
        # Ölçeklenmiş veri
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # DÖNÜŞ: X, y ve KULLANILAN SCALER
        return X_scaled, y, scaler