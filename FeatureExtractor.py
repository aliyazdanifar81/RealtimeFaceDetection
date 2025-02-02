import cv2


def face_embedding(image, model_path: str):
    recognizer = cv2.FaceRecognizerSF.create(model=model_path, config="")
    embedding = recognizer.feature(image)
    return recognizer, embedding
