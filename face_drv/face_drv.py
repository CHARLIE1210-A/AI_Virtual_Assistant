import cv2
import torch
from ultralytics import YOLO
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
import sqlite3


class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_faces(self, image):
        results = self.model(image)
        faces = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                if self.model.names[class_id] == "person" and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    faces.append((x, y, w, h))
        return faces

class RetinaFaceDetector:
    # RetinaFace detection
    def detect_faces_retina(image):
        faces = RetinaFace.detect_faces(image)
        result = []
        if faces:
            for key in faces.keys():
                identity = faces[key]
                facial_area = identity['facial_area']
                result.append(facial_area)
        return result

class FaceNetEmbedder:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])

    def get_embedding(self, face):
        face = self.transform(face).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(face).detach().numpy()
        return embedding.flatten()

class FaceVerifier:
    def __init__(self, db_path):
        # Initialize the connection to the SQLite database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def cosine_similarity(self,embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    def verify_face(self, embedding, threshold=0.6):
        # Execute a query to fetch all users' names and embeddings from the database
        self.cursor.execute('SELECT name, embedding FROM users')
        rows = self.cursor.fetchall()

        # Iterate over each stored face embedding to find a match
        for row in rows:
            stored_name, stored_embedding = row
            # Convert stored embedding from binary to numpy array
            stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)
            # Calculate the cosine similarity between the given embedding and stored embedding
            similarity = self.cosine_similarity(embedding, stored_embedding)
            # If the similarity is above the threshold, return the matched name and similarity
            if similarity > threshold:
                return stored_name, similarity
        # Return None if no match is found
        return None, None

    def close_connection(self):
        # Close the database connection
        self.conn.close()