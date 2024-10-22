import pandas as pd

# CSV dosyasını oku
veri_seti = pd.read_csv('veriseti.csv')

# İlk birkaç satırı görmek için
print(veri_seti.head())
