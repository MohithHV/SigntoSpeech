# utils/landmarks.py
import numpy as np

def extract_hand_features(hand_landmarks):
    """
    Convert mediapipe hand landmarks into a flat normalized feature vector.
    hand_landmarks: list of 21 landmarks each with x,y,z normalized to image.
    Returns a 63-length vector: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20] normalized
    relative to wrist (landmark 0) and scaled by hand size.
    """
    # Convert to numpy array
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

    # Use wrist (0) as origin
    origin = pts[0].copy()
    pts -= origin

    # Scale: use max distance from wrist to other landmarks to normalize scale
    dists = np.linalg.norm(pts, axis=1)
    scale = dists.max()
    if scale < 1e-6:
        scale = 1.0
    pts /= scale

    return pts.flatten().tolist()
