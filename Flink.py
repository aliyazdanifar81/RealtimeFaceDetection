from pyflink.datastream.functions import MapFunction, RuntimeContext
from FaceDetection import face_detector
from CropFaces import crop_faces
from FeatureExtractor import face_embedding
from Similarity import cosine_sim
import cv2


class FaceDetectionMapFunction(MapFunction):
    def __init__(self, video_path, detection_interval=10):
        self.video_path = video_path
        self.detection_interval = detection_interval  # Run face detection every N frames
        self.cap = None
        self.face_detection_model = None
        self.face_recognition_model = None
        self.trackers = []  # List to store active trackers
        self.frame_counter = 0

    def open(self, runtime_context: RuntimeContext):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        self.face_detection_model = 'models/face_detection_yunet_2023mar_int8.onnx'
        self.face_recognition_model = 'models/face_recognition_sface_2021dec_int8.onnx'

    def map(self, inp):
        while True:
            # Read the next frame
            ret, frame = self.cap.read()
            if not ret:
                break  # Exit the loop when the video ends

            # Run face detection every `detection_interval` frames
            if self.frame_counter % self.detection_interval == 0:
                # Clear existing trackers
                self.trackers = []
                # Detect faces
                faces = face_detector(frame, self.face_detection_model)
                # Initialize trackers for detected faces
                for (x, y, w, h) in faces:
                    tracker = cv2.TrackerKCF_create()  # Choose a tracker
                    bbox = (x, y, w, h)
                    tracker.init(frame, bbox)
                    self.trackers.append(tracker)

            # Update trackers for existing faces
            updated_faces = []
            for tracker in self.trackers:
                success, bbox = tracker.update(frame)
                if success:
                    updated_faces.append(bbox)  # Add the updated bounding box

            # Draw bounding boxes on the frame
            for (x, y, w, h) in updated_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

            # Save the frame with bounding boxes
            cv2.imwrite(f'./frames/{self.frame_counter}.png', frame)
            self.frame_counter += 1

            # Crop faces
            cropped_faces = crop_faces(frame, updated_faces)
            if cropped_faces:
                print(f'{self.frame_counter} - {len(cropped_faces)}')
            else:
                print(f'{self.frame_counter} - No face DETECTED!')

        return "End of video"

    def close(self):
        if self.cap:
            self.cap.release()