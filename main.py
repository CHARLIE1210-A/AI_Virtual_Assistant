import cv2
from database.database import DatabaseManager
from face_drv.face_drv import YOLOv8Detector ,FaceNetEmbedder ,FaceVerifier
from face_drv.face_processor import FaceProcessor

db_manager = DatabaseManager()
face_detector = YOLOv8Detector()
face_embedder = FaceNetEmbedder()
face_verifier = FaceVerifier('face_data.db')

# created an instance of FaceProcessor
processor = FaceProcessor(face_detector, face_embedder, face_verifier, db_manager)

# Function to start the webcam and perform real-time detection
def start_webcam():
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Discard initial frames for stabilization
    for _ in range(10):
        cap.read()   
           
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame for face detection and recognition
        processor.process_frames(frame)

        # Display the frame
        cv2.imshow("Real-Time Face Detection & Recognition", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()    
    
if __name__ == "__main__":
    image_path = "assets/images/IMG_0133.JPG" 
    image = cv2.imread(image_path)
    
    processor.process_faces(image)
    # start_webcam()

# Close the SQLite connection when done
db_manager.close()
