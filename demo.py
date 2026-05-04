import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from detector.boundary import StrokeCycleBoundaryPolicy
from viz.plot import plot_boundaries

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- tunable thresholds ---
LANDMARK_VISIBILITY_THRESHOLD = 0.7
WRIST_VISIBILITY_THRESHOLD = 0.05
YOLO_CONFIDENCE = 0.3
SMOOTH_WINDOW = 5
# --------------------------

def interpolate_nones(values):
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    nans = np.isnan(arr)
    if nans.all():
        return values
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr.tolist()

def median_smooth(values, window=5):
    arr = np.array(values, dtype=float)
    smoothed = arr.copy()
    half = window // 2
    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        window_vals = arr[start:end]
        smoothed[i] = np.nanmedian(window_vals)
    return smoothed.tolist()

yolo = YOLO('models/yolov8n.pt')
policy = StrokeCycleBoundaryPolicy(debounce_frames=15)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/pose_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.01,
    min_pose_presence_confidence=0.01,
    min_tracking_confidence=0.01
)

cap = cv2.VideoCapture("data/freestyle-video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

frames = []
right_wrist_x = []
right_shoulder_x = []
boundaries = []

prev_right = None

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # --- YOLO: detect swimmer and crop ---
        yolo_results = yolo(frame, classes=[0], conf=YOLO_CONFIDENCE, verbose=False)
        if yolo_results[0].boxes:
            box = yolo_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cropped = frame[y1:y2, x1:x2]
        else:
            right_wrist_x.append(None)
            right_shoulder_x.append(None)
            frames.append(frame_idx)
            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
            frame_idx += 1
            continue
        # -------------------------------------

        if cropped.size == 0:
            right_wrist_x.append(None)
            right_shoulder_x.append(None)
            frames.append(frame_idx)
            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / fps))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            ch, cw = cropped.shape[:2]

            right_wrist = landmarks[16]
            right_shoulder = landmarks[12]

            curr_right = None

            if right_wrist.visibility > WRIST_VISIBILITY_THRESHOLD:
                curr_right = {'wrist_x': right_wrist.x, 'shoulder_x': right_shoulder.x}
                right_wrist_x.append(right_wrist.x)
            else:
                right_wrist_x.append(None)

            right_shoulder_x.append(right_shoulder.x)

            boundary_fired = False
            if policy.is_boundary(prev_right, curr_right, frame_idx):
                boundaries.append(frame_idx)
                boundary_fired = True

            prev_right = curr_right

            for lm in landmarks:
                if lm.visibility > LANDMARK_VISIBILITY_THRESHOLD:
                    cx, cy = int(lm.x * cw), int(lm.y * ch)
                    cv2.circle(cropped, (cx, cy), 4, (0, 255, 0), -1)

            if right_wrist.visibility > WRIST_VISIBILITY_THRESHOLD:
                cv2.circle(cropped, (int(right_wrist.x * cw), int(right_wrist.y * ch)), 8, (0, 165, 255), -1)
            cv2.circle(cropped, (int(right_shoulder.x * cw), int(right_shoulder.y * ch)), 8, (255, 0, 255), -1)

            frame[y1:y2, x1:x2] = cropped

            if boundary_fired:
                cv2.putText(frame, "BOUNDARY DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        else:
            right_wrist_x.append(None)
            right_shoulder_x.append(None)

        frames.append(frame_idx)
        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print(f"frames: {len(frames)}")
print(f"right_wrist_x: {len(right_wrist_x)}")
print(f"right_shoulder_x: {len(right_shoulder_x)}")
print(f"non-None wrist: {sum(1 for x in right_wrist_x if x is not None)}")
print(f"non-None shoulder: {sum(1 for x in right_shoulder_x if x is not None)}")

# --- interpolate and smooth ---
right_wrist_x = interpolate_nones(right_wrist_x)
right_shoulder_x = interpolate_nones(right_shoulder_x)
right_wrist_x = median_smooth(right_wrist_x, window=SMOOTH_WINDOW)
right_shoulder_x = median_smooth(right_shoulder_x, window=SMOOTH_WINDOW)
# ------------------------------

plot_boundaries(frames, right_wrist_x, right_shoulder_x, boundaries, show_scatter=False)