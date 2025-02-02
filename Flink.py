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
        self.trackers = []  # stores dicts: {'tracker': tracker, 'tag': tag}
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
                self.trackers = []  # Reset trackers
                faces = face_detector(frame, self.face_detection_model)
                embeddings = []
                valid_faces = []
                for (x, y, w, h) in faces:
                    cropped_face = frame[y:y + h, x:x + w]
                    if cropped_face.size == 0:
                        continue
                    embedding = np.array(face_embedding(cropped_face, self.face_recognition_model)[1])
                    embeddings.append(embedding.flatten().tolist())
                    valid_faces.append((x, y, w, h))

                # Batch search embeddings
                tags = self.milvus_db.batch_search(embeddings, threshold=0.3)
                new_embeddings = []
                new_tags = []
                for i, tag in enumerate(tags):
                    if tag is None:
                        new_tag = self.milvus_db.assign_new_tag()
                        new_embeddings.append(embeddings[i])
                        new_tags.append(new_tag)
                        tags[i] = new_tag
                # Batch insert new faces
                if new_embeddings:
                    self.milvus_db.batch_insert(new_embeddings, new_tags)

                # Initialize trackers with tags
                for (x, y, w, h), tag in zip(valid_faces, tags):
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    self.trackers.append({'tracker': tracker, 'tag': tag})

            # Update trackers for existing faces
            updated_faces = []
            for tracker_info in self.trackers:
                tracker = tracker_info['tracker']
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    updated_faces.append((x, y, w, h, tracker_info['tag']))

            # Draw using stored tags
            for (x, y, w, h, tag) in updated_faces:
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