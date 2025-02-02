from pyflink.datastream.functions import MapFunction, RuntimeContext
from FaceDetection import face_detector
from CropFaces import crop_faces
from FeatureExtractor import face_embedding
from Similarity import cosine_sim
import cv2


class FaceDetectionMapFunction(MapFunction):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.face_detection_model = None
        self.face_recognition_model = None

    def open(self, runtime_context: RuntimeContext):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        self.face_detection_model = 'models/face_detection_yunet_2023mar_int8.onnx'
        self.face_recognition_model = 'models/face_recognition_sface_2021dec_int8.onnx'

    def map(self, inp):
        frame_counter = 0
        while True:
            # Read the next frame
            ret, frame = self.cap.read()
            if not ret:
                break  # Exit the loop when the video ends

            # Detect faces
            faces = face_detector(frame, self.face_detection_model)

            # Draw bounding boxes on the frame
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

            # Save the frame with bounding boxes
            cv2.imwrite(f'./frames/{frame_counter}.png', frame)
            frame_counter += 1

            # Crop faces
            cropped_faces = crop_faces(frame, faces)
            if cropped_faces:
                print(f'{frame_counter} - {len(cropped_faces)}')
            else:
                print(f'{frame_counter} - No face DETECTED!')

        return "End of video"

    def close(self):
        if self.cap:
            self.cap.release()
