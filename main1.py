import sys
import os
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
import cv2
import time
import collections
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from app.core.emotion_detector import create_emotion_detector

def main():
    DETECT_SECONDS = 5          # auto-close after this many seconds
    model_path = os.path.join(os.path.dirname(__file__), "emotion_model.h5")
    emotion_detector = create_emotion_detector(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with open("detected_emotion.txt", "w") as f:
            f.write("none")
        return

    emotion_counts = collections.Counter()
    start_time = time.time()
    remaining  = DETECT_SECONDS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed   = time.time() - start_time
        remaining = max(0, DETECT_SECONDS - elapsed)

        frame_with_emotion, emotion = emotion_detector.detect_emotion(frame)

        if emotion:
            emotion_counts[emotion] += 1

        # ── countdown overlay ──────────────────────────────────────────────
        h, w = frame_with_emotion.shape[:2]

        # dark bar at top
        cv2.rectangle(frame_with_emotion, (0, 0), (w, 60), (20, 20, 20), -1)

        # title
        cv2.putText(frame_with_emotion, "Vibely - Emotion Detection",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 170, 255), 2)

        # countdown
        countdown_text = f"Auto-close in: {int(remaining) + 1}s"
        cv2.putText(frame_with_emotion, countdown_text,
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 100), 2)

        # progress bar
        progress_w = int((elapsed / DETECT_SECONDS) * w)
        cv2.rectangle(frame_with_emotion, (0, 58), (progress_w, 63), (124, 58, 237), -1)
        cv2.rectangle(frame_with_emotion, (0, 58), (w, 63), (60, 60, 60), 1)

        # current emotion label (bottom)
        if emotion:
            cv2.rectangle(frame_with_emotion, (0, h - 45), (w, h), (20, 20, 20), -1)
            cv2.putText(frame_with_emotion, f"Detected: {emotion}",
                        (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 100), 2)

        cv2.imshow("Vibely - Emotion Detector", frame_with_emotion)
        cv2.setWindowProperty("Vibely - Emotion Detector", cv2.WND_PROP_TOPMOST, 1)

        # quit early with Q, or auto-close after DETECT_SECONDS
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or elapsed >= DETECT_SECONDS:
            break

    cap.release()
    cv2.destroyAllWindows()

    # save the most frequent emotion detected
    if emotion_counts:
        best_emotion = emotion_counts.most_common(1)[0][0].lower()
    else:
        best_emotion = "none"

    with open(os.path.join(os.path.dirname(__file__), "detected_emotion.txt"), "w") as f:
        f.write(best_emotion)

if __name__ == "__main__":
    main()
