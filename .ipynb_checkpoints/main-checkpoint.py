import cv2
import torch
import numpy as np
import sqlite3
from retinaface import RetinaFace
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# Create or connect to SQLite database
conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()

# Create a table for storing face details
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    description TEXT,
    image_url TEXT,
    embedding BLOB
)
''')
conn.commit()

# Load the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

def detect_faces_yolov8(image):
    # Perform object detection
    results = yolo_model(image)

    # Get image dimensions
    height, width, _ = image.shape

    # List to store face coordinates
    faces = []

    # Iterate through detected objects
    for result in results:
        for box in result.boxes:
            # Extract class ID and confidence
            class_id = int(box.cls)
            confidence = float(box.conf)

            # Check if the detected object is a person with confidence > 0.5
            if yolo_model.names[class_id] == "person" and confidence > 0.5:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                faces.append((x, y, w, h))

    return faces

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

# Function to store face details in SQLite
def store_face_details(name, age, description, image, embedding):
    cursor.execute('''
    INSERT INTO users (name, age, description, image_url, embedding)
    VALUES (?, ?, ?, ?, ?)
    ''', (name, age, description, "", embedding))  # Image URL is empty for now
    conn.commit()

# Function to convert numpy array to binary for SQLite
def convert_array_to_binary(array):
    return array.tobytes()



# Initialize Inception Resnet V1 for face embedding
face_embedding_mode = InceptionResnetV1(pretrained='vggface2').eval()

# Function to extract face embeddings
# Function to extract face embeddings
def get_face_embedding(face):
     """
    Function to extract the face embedding using InceptionResNetV1 with proper normalization.

    Parameters:
    - face: A cropped face image in BGR format (loaded using OpenCV).

    Returns:
    - A 512-dimensional embedding vector as a flattened numpy array.
    """
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    face = cv2.resize(face, (160, 160)) # Resize to the input size expected by the model
    face = torch.FloatTensor(face).permute(2, 0, 1)  # Convert to Tensor and rearrange dimensions
    face = face.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = face_embedding_model(face).detach().numpy()  # Get the face embedding
    return embedding.flatten()  # Flatten to a 1D array    


# Main function
def main(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Gather additional details
    name = input("Enter Name: ")
    age = int(input("Enter Age: "))
    description = input("Enter Description: ")
    
    # Detect faces using YOLO 
    faces = detect_faces_yolov8(image)
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        embedding = get_face_embedding(face)  # Extract face embedding using a recognition model

        # Store face details in SQLite
        store_face_details(name, age, description, face, convert_array_to_binary(embedding))
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the output
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "images/IMG_0018.JPG"  # Specify the path to your image
    main(image_path)

# Close the SQLite connection when done
conn.close()
