import cv2
import os
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# === YOLOv8 modeli yükle ===
model_yolo = YOLO("yolov8n.pt")

# === ReID'siz (torchreid gerektirmeyen) tracker ===
tracker = DeepSort(
    max_age=60,
    n_init=2,
    max_cosine_distance=0.2,
    embedder="mobilenet",
    embedder_gpu=True
)
# === Video yükle ===
cap = cv2.VideoCapture("people.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = frame_width * frame_height

# === Çıktı klasörü ve video oluştur ===
os.makedirs("output", exist_ok=True)
output = cv2.VideoWriter("output/output_stable_ids.avi",
                         cv2.VideoWriter_fourcc(*'XVID'),
                         30.0,
                         (frame_width, frame_height))

# === Filtreleme için alan oranları ===
min_area_ratio = 0.01
max_area_ratio = 0.3

# === ID için renk ve iz takibi ===
id_colors = {}
def get_color(track_id):
    if track_id not in id_colors:
        random.seed(track_id)
        id_colors[track_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return id_colors[track_id]

trajectories = defaultdict(list)
last_seen_frame = {}
frame_count = 0
max_disappear_frames = 15

# === Frame'leri işle ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

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

    if detections:
     tracks = tracker.update_tracks(detections, frame=frame)
    else:
     tracks = []

    active_ids = set()

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        color = get_color(track_id)

        # Kutuyu ve ID'yi çiz
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Merkez noktayı ekle
        center = (int((l + r) / 2), int((t + b) / 2))
        trajectories[track_id].append(center)
        last_seen_frame[track_id] = frame_count
        active_ids.add(track_id)

    # Kadrajdan çıkalı çok olanları sil
    ids_to_delete = [tid for tid, last_seen in last_seen_frame.items()
                     if frame_count - last_seen > max_disappear_frames]
    for tid in ids_to_delete:
        trajectories.pop(tid, None)
        last_seen_frame.pop(tid, None)

    # İz çiz
    for track_id, points in trajectories.items():
        if len(points) < 2:
            continue
        color = get_color(track_id)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 2)

    output.write(frame)

# === Temizlik işlemleri ===
cap.release()
output.release()
cv2.destroyAllWindows()

print("✅ Sabit ID'ler ve sadece kadrajdayken iz çizildi → output/output_stable_ids.avi")
