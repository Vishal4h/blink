import cv2
import mediapipe as mp
from blink_detector import BlinkDetector

LOW_BPM = 12
CAL_DONE_TIME = None
bpm_history = []


mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
import time
cal_start = time.time()
cal_ears = []



blink = BlinkDetector()
blinks = 0
cap = cv2.VideoCapture(0)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        h, w = frame.shape[:2]
        lm = result.multi_face_landmarks[0].landmark

        LEFT = [33, 160, 158, 133, 153, 144]
        RIGHT = [362, 385, 387, 263, 373, 380]

        left_eye = [(int(lm[i].x*w), int(lm[i].y*h)) for i in LEFT]
        right_eye = [(int(lm[i].x*w), int(lm[i].y*h)) for i in RIGHT]

        ear, blinks = blink.update(left_eye, right_eye)
        if time.time() - cal_start < 20:
            cal_ears.append(ear)
        else:
            if blink.threshold == 0.21 and cal_ears:
                blink.threshold = sum(cal_ears) / len(cal_ears) * 0.75

        


    cv2.putText(frame, f"Blinks: {blinks}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    bpm = int(blinks / max(1, (time.time() - start_time) / 60))
    bpm_history.append(bpm)
    if len(bpm_history) > 10:
        bpm_history.pop(0)
    bpm = sum(bpm_history) // len(bpm_history)

    cv2.putText(frame, f"Blinks/min: {bpm}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    
    if bpm < LOW_BPM:
                cv2.putText(frame, "BLINK MORE!", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    now = time.time()

    if now - cal_start < 20:
        status = "Calibrating"
    elif CAL_DONE_TIME is None:
        status = "Calibrated"
        CAL_DONE_TIME = now
    elif now - CAL_DONE_TIME < 5:
        status = "Calibrated"
    else:
        status = ""


    cv2.putText(frame, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)



    cv2.imshow("Blink", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
