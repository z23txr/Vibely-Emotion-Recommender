import cv2
import numpy as np
from tensorflow.keras.models import load_model
from app.core.face_detector import FaceDetector

class EmotionDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)
        self.emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
        self.face_detector = FaceDetector()

    def detect_emotion(self, frame):
        faces = self.face_detector.detect_faces(frame)
        emotion = None

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))
            prediction = self.model.predict(face)
            emotion = self.emotion_labels[np.argmax(prediction[0])]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame, emotion

def create_emotion_detector(model_path):
    return EmotionDetector(model_path)
