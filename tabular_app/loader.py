import pandas as pd

def load_data(path):
    """
    CSV dosyasını okur ve ham veriyi döndürür.
    """
    try:
        df = pd.read_csv(path)
        # Sütun isimlerindeki gereksiz boşlukları temizleyelim
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        return None