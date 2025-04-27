import cv2
import os
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# === YOLOv8 modelini yükle ===
model_yolo = YOLO("yolov8n.pt")

# === DeepSORT tracker ===
tracker = DeepSort(
    max_age=90,
    n_init=3,
    max_cosine_distance=0.3,
    embedder="mobilenet",
    half=True
)

# === Video ayarları ===
cap = cv2.VideoCapture("couple.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = frame_width * frame_height

os.makedirs("output", exist_ok=True)
output = cv2.VideoWriter("output/output_trajectory.avi",
                         cv2.VideoWriter_fourcc(*'XVID'),
                         30.0,
                         (frame_width, frame_height))

# === Alan sınırları ===
min_area_ratio = 0.01
max_area_ratio = 0.3

# === Her ID için geçmiş pozisyonları ve renkleri ===
track_history = defaultdict(list)
id_colors = {}

def get_color(id):
    if id not in id_colors:
        random.seed(id)  # Her ID için sabit renk
        id_colors[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return id_colors[id]

MAX_HISTORY = 100  # Sadece son 50 noktayı tutacağız

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model_yolo(frame)[0]
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if int(class_id) == 0 and score > 0.4:
            width = x2 - x1
            height = y2 - y1
            area = width * height
            area_ratio = area / frame_area

            if min_area_ratio < area_ratio < max_area_ratio:
                bbox = [x1, y1, width, height]
                detections.append((bbox, score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        # Kutuyu ve ID'yi çiz
        color = get_color(track_id)
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Takip izini kaydet (merkez)
        center_x = int((l + r) / 2)
        center_y = int((t + b) / 2)
        track_history[track_id].append((center_x, center_y))

        # En fazla son 50 noktayı tut
        if len(track_history[track_id]) > MAX_HISTORY:
            track_history[track_id] = track_history[track_id][-MAX_HISTORY:]

        # Geçmiş noktaları çiz
        points = track_history[track_id]
        for j in range(1, len(points)):
            if points[j - 1] is None or points[j] is None:
                continue
            cv2.line(frame, points[j - 1], points[j], color, 2)  # ID'ye özgü renkli çizgi

    output.write(frame)

cap.release()
output.release()
cv2.destroyAllWindows()

print("50 noktada sınırlı yol çizgileri 'output/output_trajectory.avi' dosyasına kaydedildi!")