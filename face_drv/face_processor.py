import cv2

class FaceProcessor:
    def __init__(self, face_detector, face_embedder, face_verifier, db_manager, window_size=(800, 600)):
        self.face_detector = face_detector  # YOLO model for face detection
        self.face_embedder = face_embedder  # Model for face embeddings
        self.face_verifier = face_verifier  # Class handling face verification
        self.db_manager = db_manager        # Database manager for storing face data
        self.window_width, self.window_height = window_size

    def resize_image_to_fit_window(self, image):
        # Get the dimensions of the image
        original_height, original_width = image.shape[:2]
        
        # Calculate the scaling factor to maintain the aspect ratio
        scale_width = self.window_width / original_width
        scale_height = self.window_height / original_height
        scale = min(scale_width, scale_height)
        
        # Calculate the new dimensions of the image
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image

    def process_faces(self, image):
        """Process faces from an image."""
        # Resize image to fit window
        image = self.resize_image_to_fit_window(image)

        # Detect faces using YOLO
        faces = self.face_detector.detect_faces(image)

        # Process each detected face
        for (x1, y1, x2, y2) in faces:
            face = image[y1:y2, x1:x2]
            self.process_single_face(face, (x1, y1, x2, y2), image)

        # Display the output
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_frames(self, frame):
        """Process faces from a webcam or video feed frame."""
        # Detect faces using YOLO
        faces = self.face_detector.detect_faces(frame)

        # Process each detected face
        for (x1, y1, x2, y2) in faces:
            face = frame[y1:y2, x1:x2]
            self.process_single_face(face, (x1, y1, x2, y2), frame)

        # No need to destroy windows here as it is used for real-time processing

    def process_single_face(self, face, bbox, image):
        """Process a single face: recognize or store in the database."""
        x1, y1, x2, y2 = bbox
        embedding = self.face_embedder.get_embedding(face)

        if embedding is None:
            return

        # Verify the face against the stored embeddings in the database
        matched_name, similarity = self.face_verifier.verify_face(embedding)
        if matched_name:
            label = f"{matched_name} ({similarity:.2f})"
            print(f"Face matched with: {matched_name}, Similarity: {similarity}")
        else:
            label = self.store_new_face(face, embedding)

        # Draw rectangle and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def store_new_face(self, face, embedding):
        """Store a new face in the database with user input."""
        print("New face detected. Storing information...")
        name = input("Enter Name: ")
        age = int(input("Enter Age: "))
        description = input("Enter Description: ")
        image_url = "image_url"  # Placeholder; you can store the image somewhere and get the URL
        self.db_manager.store_face_details(name, age, description, image_url, embedding.tobytes())
        print(f"Stored face details for {name}")
        return name

# Example Usage
# You would have classes for face detection, embedding, verification, and database management.
# For example:
# face_detector = YOLOFaceDetector()
# face_embedder = FaceNetEmbedder()
# face_verifier = FaceVerifier(db_manager)
# db_manager = SQLiteManager()

# Then create an instance of FaceProcessor
# processor = FaceProcessor(face_detector, face_embedder, face_verifier, db_manager)

# Use processor to process a single image or real-time webcam feed
# processor.process_faces(image)
# processor.process_frames(frame)
