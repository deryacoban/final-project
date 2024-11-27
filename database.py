import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resim yolu
image_path = "C:\\Users\\derya\\OneDrive\\Belgeler\\GitHub\\final-project\\images.jpg"

# Yaş tahmini modeli dosyaları
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

# Yaş aralıkları
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Modeli yükle
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

# Haar Cascade ile yüz tespiti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Resmi yükle
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yüzleri tespit et
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))


# Her yüz için yaş tahmini
for (x, y, w, h) in faces:
    # Yüzü çıkar
    face_img = image[y:y+h, x:x+w]
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Yaş tahmini yap
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    # Bounding box çiz ve yaş yaz
    label = f"Age: {age}"
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Sonuçları göster
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

print(f"Detected faces: {faces}")
