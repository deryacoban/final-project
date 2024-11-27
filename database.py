import os
import requests

# Pixabay API anahtarınızı buraya girin
API_KEY = '47297716-940ad03c7c69f9ee90773ca31'
SEARCH_QUERY = 'human'  # Arama kriteri (örneğin: "insan")
URL = f"https://pixabay.com/api/?key={API_KEY}&q={SEARCH_QUERY}&image_type=photo"

# Kaydetme klasörünüzü tanımlayın
save_dir = r"C:\\Users\\Derya\\OneDrive\\Masaüstü\\Yeni klasör"

# Klasör mevcut değilse oluştur
os.makedirs(save_dir, exist_ok=True)

# API'yi çağır ve sonuçları al
response = requests.get(URL)

if response.status_code == 200:
    data = response.json()
    photos = data.get("hits", [])

    if photos:
        print(f"{len(photos)} fotoğraf bulundu.")
        # Fotoğrafları indir ve belirtilen klasöre kaydet
        for i, photo in enumerate(photos[:5]):  # İlk 5 fotoğrafı indirir
            image_url = photo['largeImageURL']
            response = requests.get(image_url)

            if response.status_code == 200:
                # Dosya adını ve yolunu oluştur
                filename = os.path.join(save_dir, f"photo_{i+1}.jpg")
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"{filename} başarıyla kaydedildi.")
            else:
                print(f"{image_url} indirilemedi.")
    else:
        print("Hiçbir fotoğraf bulunamadı.")
else:
    print(f"Hata oluştu: {response.status_code}")
