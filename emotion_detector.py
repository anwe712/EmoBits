import cv2
import numpy as np
import pygame
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Initialize pygame
pygame.mixer.init()

# Load face detector and emotion model
face_classifier = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
classifier = load_model("models/model.h5")

# Emotion labels and corresponding music files
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
music_files = {
    'Angry': "music/angry.mp3",
    'Disgust': "music/neutral.mp3",
    'Fear': "music/neutral.mp3",
    'Happy': "music/happy.mp3",
    'Neutral': "music/neutral.mp3",
    'Sad': "music/happy.mp3",  
    'Surprise': "music/happy.mp3"
}

current_emotion = None  # Track last played emotion

def detect_emotion(image):
    global current_emotion
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected"

    x, y, w, h = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    roi = roi_gray.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    prediction = classifier.predict(roi)[0]
    detected_emotion = emotion_labels[prediction.argmax()]

    if detected_emotion != current_emotion:
        pygame.mixer.music.stop()  
        pygame.mixer.music.load(music_files.get(detected_emotion, ""))
        pygame.mixer.music.play()
        current_emotion = detected_emotion

    return detected_emotion
