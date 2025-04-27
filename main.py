import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")

# DeepSORT tracker (ReID için optimize edildi)
tracker = DeepSort(
    max_age=60,                     # ID kaybolmadan önce 60 frame bekle
    n_init=3,                       # Bir nesne 3 frame arka arkaya görülürse ID ver
    max_cosine_distance=0.3,        # Biraz esnek eşleştirme
    embedder="osnet_x0_25",         # ReID için daha iyi bir embedder (person için optimize)
    half=True                       # GPU varsa daha hızlı
)

# Video okuma ve çıkış ayarları
cap = cv2.VideoCapture("couple.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("output", exist_ok=True)
output = cv2.VideoWriter("output/output.avi",
                         cv2.VideoWriter_fourcc(*'XVID'),
                         30.0,
                         (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # İnsan tespiti
    results = model(frame)[0]
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if int(class_id) == 0 and score > 0.4:  # Sadece insan ve 0.4'ten büyük skor
            bbox = [x1, y1, x2 - x1, y2 - y1]   # x, y, w, h
            detections.append((bbox, score, 'person'))

    # DeepSORT ile takip
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        # Görüntüye çiz
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    output.write(frame)

cap.release()
output.release()
cv2.destroyAllWindows()
print("Video işleme tamamlandı. Çıkış dosyası: output/output.avi")