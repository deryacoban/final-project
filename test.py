#import pandas as pd

"""# CSV dosyasını oku
veri_seti = pd.read_sql("SELECT * FROM table_name", conn)  # SQL sorgusu ile veri çekme

# İlk birkaç satırı görmek için
print(veri_seti.head())"""

"""import pyodbc
import os

# Veritabanı bağlantı bilgileriniz
server = 'DERYA'
database = 'bitirme'
driver= '{ODBC Driver 17 for SQL Server}' # Sürücünüzün versiyonuna göre değiştirin

# Bağlantı stringi
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database}'
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

# Resim dosyasını oku
image_path = 'path_to_your_image.jpg'
with open(image_path, 'rb') as file:
    image_data = file.read()

# SQL sorgusu
sql = 'insert INTO resimler values(1, (SELECT * FROM OPENROWSET (BULK r''C:\\Users\\derya\\OneDrive\\Masaüstü\\train\\images.jpg'', SINGLE_BLOB )as T1))'


# Resim dosyasını ve ismini veritabanına ekle
cursor.execute(sql, (os.path.basename(image_path), image_data))
conn.commit()

# Bağlantıyı kapat
cursor.close()
conn.close()

print("Resim başarıyla yüklendi.")"""
