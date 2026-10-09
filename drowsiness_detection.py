# drowsiness detection project
# uses webcam + mediapipe face mesh to track eyes and mouth
# EAR -> eyes closed too long = drowsy
# MAR -> mouth open = yawn
# also sends a telegram alert to my phone
# press q to quit

import csv
import os
import threading
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import requests

# settings - change these if detection feels off
EYE_CLOSED_SECONDS = 1.5     # eyes closed more than this = drowsy
MAR_THRESHOLD = 0.6          # above this = yawning
YAWN_SECONDS = 1.0           # mouth has to stay open this long
CALIBRATION_SECONDS = 3      # first 3 sec to measure normal eye EAR
DEFAULT_EAR_THRESHOLD = 0.22 # backup value if calibration doesn't work
LOG_FILE = "drowsiness_log.csv"

# telegram stuff (put your own token and chat id here)
TELEGRAM_BOT_TOKEN = "PASTE_YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "PASTE_YOUR_CHAT_ID_HERE"
ALERT_COOLDOWN_SECONDS = 30  # so it doesnt spam my phone

# face mesh landmark points
# eye: corner, top, top, corner, bottom, bottom
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# mouth: upper lip, lower lip, left corner, right corner
MOUTH = [13, 14, 78, 308]


def dist(a, b):
    return np.linalg.norm(a - b)


# EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
# goes close to 0 when eye closes
def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4))


# MAR = mouth height / mouth width
def mouth_aspect_ratio(pts):
    top, bottom, left, right = pts
    return dist(top, bottom) / dist(left, right)


# save events to csv
def log_event(event, value):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "event", "value"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, f"{value:.3f}"])


def telegram_configured():
    return not TELEGRAM_BOT_TOKEN.startswith("PASTE") and not TELEGRAM_CHAT_ID.startswith("PASTE")


# sends photo + msg to phone
# running it in a thread otherwise the video freezes while sending
def send_telegram_alert(frame, closed_for):
    def worker(img):
        try:
            ok, jpg = cv2.imencode(".jpg", img)
            if not ok:
                return
            caption = (f"DROWSINESS ALERT!\n"
                       f"Eyes closed for {closed_for:.1f} seconds\n"
                       f"Time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                              files={"photo": ("alert.jpg", jpg.tobytes(), "image/jpeg")}, timeout=10)
            print("Phone alert sent" if r.ok else f"Telegram error: {r.text}")
        except Exception as e:
            print(f"Could not send phone alert: {e}")

    threading.Thread(target=worker, args=(frame.copy(),), daemon=True).start()


# alarm beep - making the tone with numpy so i dont need a sound file
class Alarm:

    def __init__(self):
        self.playing = False
        self.sound = None
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            t = np.linspace(0, 0.5, int(44100 * 0.5), False)
            tone = (np.sin(2 * np.pi * 880 * t) * 32767 * 0.5).astype(np.int16)
            self.sound = pygame.sndarray.make_sound(tone)
        except Exception as e:
            print(f"[Warning] Audio unavailable, alarm will be visual only: {e}")

    def start(self):
        if self.sound and not self.playing:
            self.sound.play(loops=-1)
            self.playing = True

    def stop(self):
        if self.sound and self.playing:
            self.sound.stop()
            self.playing = False


# text with black outline so its readable
def draw_text(frame, text, pos, color, scale=0.6, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Close other apps using the camera and retry.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    alarm = Alarm()

    start_time = time.time()
    calib_values = []
    calibrated = False
    ear_threshold = DEFAULT_EAR_THRESHOLD

    eyes_closed_since = None
    drowsy_logged = False
    yawn_since = None
    yawn_counted = False

    blinks = yawns = drowsy_events = 0
    last_phone_alert = 0

    if telegram_configured():
        print("Telegram phone alerts: ON")
    else:
        print("Telegram phone alerts: OFF (add your bot token and chat ID to enable)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()

        status, color = "No face detected", (0, 165, 255)
        is_drowsy = False
        ear = mar = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            pts = np.array([(p.x * w, p.y * h) for p in lm])

            # average of both eyes
            ear = (eye_aspect_ratio(pts[LEFT_EYE]) + eye_aspect_ratio(pts[RIGHT_EYE])) / 2
            mar = mouth_aspect_ratio(pts[MOUTH])

            for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                cv2.circle(frame, tuple(pts[idx].astype(int)), 2, (0, 255, 0), -1)

            # calibration - threshold = 75% of normal open eye EAR
            if not calibrated:
                calib_values.append(ear)
                remaining = CALIBRATION_SECONDS - (now - start_time)
                status = f"Calibrating - keep eyes open ({max(remaining, 0):.1f}s)"
                color = (255, 200, 0)
                if remaining <= 0:
                    if calib_values:
                        ear_threshold = 0.75 * float(np.median(calib_values))
                    calibrated = True
                    print(f"Calibrated EAR threshold: {ear_threshold:.3f}")

            # actual detection
            else:
                status, color = "Awake", (0, 200, 0)

                # eyes
                if ear < ear_threshold:
                    if eyes_closed_since is None:
                        eyes_closed_since = now
                    closed_for = now - eyes_closed_since
                    if closed_for >= EYE_CLOSED_SECONDS:
                        is_drowsy = True
                        if not drowsy_logged:  # only once per closure
                            drowsy_events += 1
                            log_event("DROWSY", closed_for)
                            drowsy_logged = True
                            if telegram_configured() and now - last_phone_alert >= ALERT_COOLDOWN_SECONDS:
                                send_telegram_alert(frame, closed_for)
                                last_phone_alert = now
                else:
                    # short closure = just a blink
                    if eyes_closed_since is not None and (now - eyes_closed_since) < EYE_CLOSED_SECONDS:
                        blinks += 1
                    eyes_closed_since = None
                    drowsy_logged = False

                # mouth / yawn
                if mar > MAR_THRESHOLD:
                    if yawn_since is None:
                        yawn_since = now
                    if now - yawn_since >= YAWN_SECONDS:
                        status, color = "Yawning", (0, 255, 255)
                        if not yawn_counted:
                            yawns += 1
                            log_event("YAWN", mar)
                            yawn_counted = True
                else:
                    yawn_since = None
                    yawn_counted = False

                if is_drowsy:
                    status, color = "DROWSY! WAKE UP!", (0, 0, 255)

        # alarm + red screen
        if is_drowsy:
            alarm.start()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        else:
            alarm.stop()

        # on screen info
        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        draw_text(frame, status, (10, 28), color, scale=0.8)
        draw_text(frame, f"EAR: {ear:.2f}  (threshold {ear_threshold:.2f})", (10, h - 70), (255, 255, 255))
        draw_text(frame, f"MAR: {mar:.2f}  (threshold {MAR_THRESHOLD:.2f})", (10, h - 45), (255, 255, 255))
        draw_text(frame, f"Blinks: {blinks}   Yawns: {yawns}   Drowsy alerts: {drowsy_events}",
                  (10, h - 20), (255, 255, 255))

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Session summary -> Blinks: {blinks}, Yawns: {yawns}, Drowsy alerts: {drowsy_events}")
    print(f"Events saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
