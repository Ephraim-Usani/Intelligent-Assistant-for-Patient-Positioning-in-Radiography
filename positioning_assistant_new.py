import cv2
import mediapipe as mp
from ultralytics import YOLO
import torch
import numpy as np

# -------------------------------
# Load separate YOLO models
# -------------------------------
model_cassette = YOLO(r"C:\Users\User\Downloads\Ephraim Files\positioning assistant\YOLOv8n_cassette.pt")
model_light = YOLO(r"C:\Users\User\Downloads\Ephraim Files\positioning assistant\yolov8n_cassette_light_field_seg_final.pt")

# Force both models to run on CPU
model_cassette.to('cpu')
model_light.to('cpu')

# Allow PyTorch to use multiple threads
torch.set_num_threads(6)

# -------------------------------
# Initialize MediaPipe Pose
# -------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Open camera
# -------------------------------
cap = cv2.VideoCapture(1)  # change to 0, 1, or 2 if needed

# -------------------------------
# Sensitivity thresholds
# -------------------------------
tilt_threshold = 0.015
rotation_threshold = 0.015
obliquity_threshold = 0.07

# -------------------------------
# Create display window
# -------------------------------
cv2.namedWindow("Chest X-ray Positioning + Collimation Assistant", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Chest X-ray Positioning + Collimation Assistant", 960, 720)

# -------------------------------
# Bounding box parameters
# -------------------------------
PIXELS_PER_CM = 20  # adjust for scale
VERTICAL_TRIM_CM = 1  # top down + bottom up adjustment

def cm_to_pixels(cm):
    return int(cm * PIXELS_PER_CM)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 720))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # MediaPipe Pose
    # -------------------------------
    results = pose.process(rgb)
    checklist = []

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        nose = lm[mp_pose.PoseLandmark.NOSE]

        # Pose checks
        back_to_camera_ok = nose.z > (ls.z + rs.z) / 2
        shoulders_level_ok = abs(ls.y - rs.y) < tilt_threshold
        mid_shoulder_x = (ls.x + rs.x) / 2
        mid_hip_x = (lh.x + rh.x) / 2
        spine_midline_ok = abs(mid_shoulder_x - mid_hip_x) < rotation_threshold
        shoulder_z_diff = abs(ls.z - rs.z)
        hip_z_diff = abs(lh.z - rh.z)
        obliquity_ok = (shoulder_z_diff < obliquity_threshold) and (hip_z_diff < obliquity_threshold)

        checklist.extend([
            ("PA", back_to_camera_ok),
            ("Shoulders Level (Tilt)", shoulders_level_ok),
            ("Spine Midline (Rotation)", spine_midline_ok),
            ("No Obliquity", obliquity_ok)
        ])

    # -------------------------------
    # YOLOv8 Detection
    # -------------------------------
    results_cassette = model_cassette.predict(frame, imgsz=640, verbose=False)
    results_light = model_light.predict(frame, imgsz=640, verbose=False)

    cassette_detected = False
    light_detected = False
    collimation_status = None

    # -------------------------------
    # Extract best cassette box
    # -------------------------------
    cassette_box = None
    for r in results_cassette:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cassette_box = (x1, y1, x2, y2)
            cassette_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Cassette {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            break

    # -------------------------------
    # Extract best light field box
    # -------------------------------
    light_box = None
    for r in results_light:
        for box in r.boxes:
            label_name = model_light.names[int(box.cls[0])].lower()
            if "light" not in label_name and "beam" not in label_name:
                continue  # skip cassette detections in the light model

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Apply vertical trimming
            trim_px = cm_to_pixels(VERTICAL_TRIM_CM)
            y1 = min(frame.shape[0] - 1, max(0, y1 + trim_px))
            y2 = min(frame.shape[0] - 1, max(0, y2 - trim_px))

            light_box = (x1, y1, x2, y2)
            light_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"Light Field {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            break

    # -------------------------------
    # Collimation check (only over-collimation)
    # -------------------------------
    if cassette_box and light_box:
        cx1, cy1, cx2, cy2 = cassette_box
        lx1, ly1, lx2, ly2 = light_box

        if lx1 < cx1 or ly1 < cy1 or lx2 > cx2 or ly2 > cy2:
            collimation_status = "Over-Collimated"
        else:
            collimation_status = "Collimated"

    # -------------------------------
    # Display checklist and status
    # -------------------------------
    y_offset = 40
    for text, passed in checklist:
        color = (0, 255, 0) if passed else (0, 0, 255)
        symbol = "[OK]" if passed else "[X]"
        cv2.putText(frame, f"{symbol} {text}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_offset += 30

    # Status display
    status_texts = []
    if cassette_detected:
        status_texts.append("Cassette Detected")
    if light_detected:
        status_texts.append("Light Field Detected")
    if collimation_status:
        status_texts.append(collimation_status)

    for status in status_texts:
        color = (0, 255, 0)
        if "Over" in status:
            color = (0, 0, 255)
        cv2.putText(frame, status, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y_offset += 35

    # Show frame
    cv2.imshow("Chest X-ray Positioning + Collimation Assistant", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
