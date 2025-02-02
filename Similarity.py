import cv2


def cosine_sim(face_feature, reference_feature, recognizer):
    cosin_score = recognizer.match(face_feature, reference_feature, cv2.FaceRecognizerSF_FR_COSINE)
    return cosin_score
