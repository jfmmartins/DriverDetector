import cv2
import numpy as np
from collections import deque
import winsound
import time


IMG_SIZE = 64
CATEGORIES = ["Open_Eyes", "Closed_Eyes"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

history = deque(maxlen=5)  # last 5 frames

eyes_closed_start = None
ALERT_SECONDS = 3

def web_video(model):
    global eyes_closed_start
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        final_label = "No face detected"

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            preds = []
            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
                eye_img = eye_img / 255.0
                eye_img = eye_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                pred = model.predict(eye_img, verbose=0)
                preds.append(pred)

            if preds:
                avg_pred = np.mean(preds, axis=0)  # average predictions
                label = CATEGORIES[np.argmax(avg_pred)]
                history.append(label)

                # Majority vote from last N frames
                final_label = max(set(history), key=history.count)

                # Draw rectangles and label
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255,0,0), 2)
                cv2.putText(frame, final_label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                if final_label=="Closed_Eyes":
                    if eyes_closed_start:
                        print(time.time()-eyes_closed_start)
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                        print("night night")
                    elif time.time()-eyes_closed_start >= ALERT_SECONDS:
                        winsound.Beep(1000,500)
                else:
                    eyes_closed_start = None

        cv2.imshow("Driver Eye State", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()