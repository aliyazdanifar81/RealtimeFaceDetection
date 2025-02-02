from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from Similarity import cosine_sim
import cv2


class MilvusFaceDatabase:
    def __init__(self, recognizer_path, collection_name="face_embeddings", embedding_dim=128):
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None
        self.next_tag_id = 1  # Counter for assigning new tags

    def connect(self, host="localhost", port="19530"):
        """Connect to Milvus server."""
        connections.connect("default", host=host, port=port)
        print("Connected to Milvus.")

    def create_collection(self):
        """Create a collection for storing face embeddings."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=50)
        ]
        schema = CollectionSchema(fields, "Face Embeddings Collection")
        self.collection = Collection(self.collection_name, schema)
        print(f"Collection '{self.collection_name}' created.")

    def create_index(self):
        """Create an index for the embedding field."""
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)
        print("Index created.")

    def batch_insert(self, embeddings, tags):
        """Insert multiple face embeddings and tags into the collection in a batch."""
        if not self.collection:
            raise RuntimeError("Collection not initialized.")

        # Ensure the input lists have the same length
        if len(embeddings) != len(tags):
            raise ValueError("The number of embeddings and tags must be the same.")

        # Prepare the data for insertion
        data = [
            embeddings,  # List of embeddings
            tags  # List of tags
        ]

        # Insert the data into the collection
        self.collection.insert(data)
        self.collection.flush()  # Ensure data persistence
        print(f"Inserted {len(embeddings)} faces into the collection.")

    def batch_search(self, embeddings, threshold=0.5):
        """Search for multiple faces in the collection using cosine similarity."""
        if not self.collection:
            raise RuntimeError("Collection not initialized.")

        # Load the collection into memory
        self.collection.load()

        # Retrieve all embeddings and tags from the collection
        results = self.collection.query(
            expr='',
            output_fields=["embedding", "tag"],
            limit=1000  # Adjust the limit based on your collection size
        )

        # List to store the best match for each query embedding
        tags = []

        # Compare each query embedding with all retrieved embeddings
        for embedding in embeddings:
            best_match = None
            best_score = -1  # Initialize with the lowest possible score

            for result in results:
                reference_embedding = result["embedding"]
                tag = result["tag"]

                # Compute cosine similarity
                score = cosine_sim(embedding, reference_embedding, self.recognizer)

                # Check if this is the best match so far
                if score > best_score:
                    best_score = score
                    best_match = tag

            # Apply the threshold to determine if a match exists
            if best_score >= threshold:
                tags.append(best_match)
            else:
                tags.append(None)  # No match found

        return tags

    def release(self):
        """Release the collection."""
        if self.collection:
            self.collection.release()
            print("Collection released.")

    def assign_new_tag(self):
        """Generate a new tag for an unknown face."""
        tag = f"Person_{self.next_tag_id}"
        self.next_tag_id += 1
        return tag
