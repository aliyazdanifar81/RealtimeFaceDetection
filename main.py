from pyflink.datastream import StreamExecutionEnvironment
from Flink import FaceDetectionMapFunction
from FaceDetection import face_detector
import cv2
from CropFaces import crop_faces

# Set up the Flink environment
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)

# Open the video file
video_path = "./videos/Terminal1.mp4"

# Add the frames to the Flink stream
env.from_collection([0]).map(FaceDetectionMapFunction(video_path)).print()

# Execute the Flink job
env.execute("Real-time Face Detection")
