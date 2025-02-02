import numpy as np
import cv2

def cosine_sim(face_feature, reference_feature, recognizer):
    # Ensure features are numpy arrays
    if not isinstance(face_feature, np.ndarray):
        face_feature = np.array(face_feature, dtype=np.float32)
    if not isinstance(reference_feature, np.ndarray):
        reference_feature = np.array(reference_feature, dtype=np.float32)

    # Use OpenCV's cosine similarity method
    cosin_score = recognizer.match(face_feature, reference_feature, cv2.FaceRecognizerSF_FR_COSINE)
    return cosin_score
