import cv2


def face_detector(image, model_path: str):
    # Initialize the face detector
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),  # Adjust input size as needed
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000  # Maximum number of faces to detect
    )

    # Set input size for the detector
    height, width, _ = image.shape
    detector.setInputSize((width, height))

    # Detect faces
    retval, faces = detector.detect(image)

    # Check if faces were detected
    if retval == 0 or faces is None:
        return []  # Return an empty list if no faces are detected

    # Extract bounding box coordinates from the detected faces
    detected_faces = []
    for face in faces:
        # face is a 1D array: [x, y, w, h, confidence, ...]
        x, y, w, h = map(int, face[:4])  # Convert to integers
        detected_faces.append((x, y, w, h))

    return detected_faces
