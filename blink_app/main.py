import cv2
import mediapipe as mp
import time
from blink_detector import BlinkDetector

LOW_BPM = 12
CALIBRATION_TIME = 20
FRAME_SKIP = 2

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

blink = BlinkDetector()
cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

start_time = time.time()
cal_start = time.time()
cal_ears = []
bpm_history = []

frame_count = 0
blinks = 0
calibrated = False

ui_blinks = 0
ui_bpm = 0
ui_status = "Calibrating"
ui_warn = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    run_detection = frame_count % FRAME_SKIP == 0

    if run_detection:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        face_visible = False

        if result.multi_face_landmarks:
            face_visible = True
            h, w = frame.shape[:2]
            lm = result.multi_face_landmarks[0].landmark

            left_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]

            ear, blinks = blink.update(left_eye, right_eye)

            if not calibrated:
                if time.time() - cal_start < CALIBRATION_TIME:
                    cal_ears.append(ear)
                    ui_status = "Calibrating"
                else:
                    if cal_ears:
                        blink.threshold = (sum(cal_ears) / len(cal_ears)) * 0.75
                    calibrated = True
                    ui_status = ""
            else:
                ui_status = ""

        else:
            ui_status = "Face Not Detected"

        elapsed_minutes = max(1e-6, (time.time() - start_time) / 60)
        bpm = int(blinks / elapsed_minutes)

        bpm_history.append(bpm)
        if len(bpm_history) > 10:
            bpm_history.pop(0)
        bpm = sum(bpm_history) // len(bpm_history)

        ui_blinks = blinks
        ui_bpm = bpm
        ui_warn = calibrated and face_visible and bpm < LOW_BPM

    y = 40
    gap = 40

    cv2.putText(frame, f"Blinks: {ui_blinks}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    y += gap
    cv2.putText(frame, f"Blinks/min: {ui_bpm}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    y += gap
    if ui_warn:
        cv2.putText(frame, "BLINK MORE!", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    y += gap
    if ui_status:
        cv2.putText(frame, ui_status, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Blink", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
