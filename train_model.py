# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_FILE = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.pkl"

def main():
    if not os.path.exists(DATA_FILE):
        print("No data found. Collect data first using app.py in collection mode.")
        return

    df = pd.read_csv(DATA_FILE)
    if 'label' not in df.columns:
        print("CSV missing 'label' column.")
        return

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print("Saved trained model to", MODEL_PATH)

if __name__ == "__main__":
    main()
