def crop_faces(image, faces: list):
    # List to store cropped faces
    cropped_faces = []

    # Check if faces are detected
    try:
        if faces is not None:
            for i, face in enumerate(faces):
                # Extract bounding box coordinates
                x, y, w, h = map(int, face[:4])

                # Crop the face from the image
                cropped_face = image[y:y + h, x:x + w]

                # Append the cropped face to the list
                cropped_faces.append(cropped_face)
            return cropped_faces
    except Exception as e:
        print(e)