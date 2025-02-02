from pyflink.datastream.functions import MapFunction, RuntimeContext
from FaceDetection import face_detector
from Milvus import MilvusFaceDatabase
import numpy as np
from CropFaces import crop_faces
from FeatureExtractor import face_embedding
import cv2


class FaceDetectionMapFunction(MapFunction):
    def __init__(self, video_path, detection_interval=10):
        self.milvus_db = None
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
        # Milvus database handler
        self.milvus_db = MilvusFaceDatabase(self.face_recognition_model)
        # Initialize Milvus
        self.milvus_db.connect()
        self.milvus_db.create_collection()
        self.milvus_db.create_index()

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

            # Process each detected face
            for (x, y, w, h) in updated_faces:
                # Crop the face
                cropped_face = frame[y:y + h, x:x + w]
                if cropped_face.size == 0:
                    continue  # Skip empty crops

                # Compute face embedding
                embedding = np.array(face_embedding(cropped_face, self.face_recognition_model)[1]).flatten().tolist()

                # Search for the face in Milvus
                tag = self.milvus_db.search_face(embedding, threshold=0.3)
                if tag is None:
                    # Assign a new tag and insert into Milvus
                    tag = self.milvus_db.assign_new_tag()
                    self.milvus_db.insert_face(embedding, tag)

                # Draw bounding box and tag
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, tag, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the frame with bounding boxes
            cv2.imwrite(f'./frames/{self.frame_counter}.png', frame)
            self.frame_counter += 1

        return "End of video"

    def close(self):
        if self.cap:
            self.cap.release()
        if self.milvus_db:
            self.milvus_db.release()
