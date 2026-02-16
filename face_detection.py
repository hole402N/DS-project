import cv2
import threading
import os
from alarm import alarm

# Get current directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct Haar Cascade path
cascade_path = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if cascade loaded properly
if face_cascade.empty():
    print("Error loading Haar Cascade file")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening webcam")
    exit()

alarm_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # If face detected
    if len(faces) > 0:
        if not alarm_triggered:
            alarm_triggered = True
            threading.Thread(target=alarm, daemon=True).start()
    else:
        alarm_triggered = False

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection Alarm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()