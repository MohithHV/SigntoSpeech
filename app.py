# app.py
import cv2
import mediapipe as mp
import joblib
import os
import csv
import time
import argparse
import pyttsx3
from utils.landmarks import extract_hand_features

# Paths
MODEL_PATH = "models/gesture_model.pkl"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "gestures.csv")

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)

def speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

def save_example(features, label):
    header = False
    if not os.path.exists(DATA_FILE):
        header = True
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if header:
            # create header: f0,f1,...,f62,label
            cols = [f"f{i}" for i in range(len(features))] + ["label"]
            writer.writerow(cols)
        writer.writerow(features + [label])

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded from", MODEL_PATH)
        return model
    else:
        print("No trained model found at", MODEL_PATH)
        return None

def main(args):
    mode = "classification"  # or 'collection'
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        last_spoken = ""
        last_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = hands.process(rgb)

            label_text = f"Mode: {mode}"
            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Feature extraction
                features = extract_hand_features(hand_landmarks.landmark)

                # If in collection mode and user pressed a key (handled below), saving occurs there
                if mode == "classification":
                    if model is not None:
                        pred = model.predict([features])[0]
                        label_text += f" | Pred: {pred}"
                        # speak once per detection with small cooldown
                        now = time.time()
                        if pred != last_spoken or (now - last_time) > 2.0:
                            speak(str(pred))
                            last_spoken = pred
                            last_time = now
                    else:
                        label_text += " | No model (press 't' to train)"
            else:
                label_text += " | No hand detected"

            # Overlay instructions and label
            cv2.putText(frame, label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, "Toggle mode: 'm' | Collect label: keys A-Z | Train: 't' | Quit: 'q'", (10,470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Sign2Speech - BodyTalk", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mode = "collection" if mode == "classification" else "classification"
                print("Mode changed to", mode)
            elif key == ord('t'):
                # Run training script externally — we will call train_model.py
                print("Training model (this may take a while)...")
                os.system("python train_model.py")
                model = load_model()
            else:
                # Data collection: if in collection mode and user presses A-Z or a-z
                if mode == "collection" and res.multi_hand_landmarks:
                    if (65 <= key <= 90) or (97 <= key <= 122):  # A-Z or a-z
                        label = chr(key).upper()
                        save_example(features, label)
                        print(f"Saved example for label '{label}'")
                        speak(label)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="classification", help="start mode: classification or collection")
    args = parser.parse_args()
    main(args)
