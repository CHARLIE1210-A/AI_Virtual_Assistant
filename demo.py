import os
import cv2
import numpy as np
from deepface import DeepFace
import keyboard
import time
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from pydub import AudioSegment
import pyaudio
import wave
import requests

genai.configure(api_key="AIzaSyCPVWzopKsAxYjwakAklSiN64wAAWA1VZo")

def text_to_speech(text, api_key, voice_id, output_path="output.mp3"):
    """
    Converts text to speech using the ElevenLabs Text-to-Speech API.

    Args:
        text (str): The text to convert to speech.
        api_key (str): Your ElevenLabs API key.
        voice_id (str): The voice ID to use for speech synthesis.
        output_path (str): The path where the audio file will be saved. Defaults to "output.mp3".

    Returns:
        str: Path to the generated audio file.

    Raises:
        Exception: If the API request fails or encounters an error.
    """
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(tts_url, headers=headers, json=data, stream=True)
        if response.ok:
            # Save the audio to the specified file
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Audio saved to {output_path}")
            return output_path
        else:
            # Raise an exception if the API request fails
            raise Exception(f"Error: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        raise Exception(f"Request failed: {e}")

def convert_and_play_audio(input_audio_path, output_audio_path="output.wav"):
    """
    Converts an audio file to WAV format and plays it.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str): Path to save the converted WAV file. Defaults to "output.wav".

    Returns:
        str: Path to the saved WAV file.

    Raises:
        Exception: If any error occurs during processing or playback.
    """
    try:
        # Convert to WAV format
        audio = AudioSegment.from_file(input_audio_path)
        audio.export(output_audio_path, format="wav")
        print(f"Audio converted and saved as {output_audio_path}")

        # Play the WAV file
        chunk = 1024  # Chunk size for reading audio

        # Open the sound file
        wf = wave.open(output_audio_path, 'rb')

        # Create an interface to PortAudio
        p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        # Read and play audio in chunks
        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        # Close and terminate the stream
        stream.close()
        p.terminate()

        print("Audio playback completed.")
        return output_audio_path

    except Exception as e:
        raise Exception(f"An error occurred: {e}")

class VisionPipeline:
    def __init__(self,database_path="db/", unknown_faces_path="stored_img/"):
        self.database_path = database_path
        self.unknown_faces_path = unknown_faces_path
        self.active_task = "normal"  # Currently active task
        self.running = True  # Flag to control the main loop
        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.unknown_faces_path, exist_ok=True)
        self.face_id = 0  # Unique ID for storing new faces
        self.query_triggered = False
        self.query_frame_count = 0
        
    def verify_face_with_database(self, detected_face_rgb, model_name="Facenet"):
        """
        Verify if a detected face matches any in the database.

        Parameters:
        - detected_face_rgb: RGB image of the detected face (preprocessed).
        - model_name: DeepFace model name to use for verification.

        Returns:
        - Tuple (face_found, matched_file_name) or (False, None) if no match.
        """
        image_files = [
            f for f in os.listdir(self.database_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not image_files:
            print("Database is empty.")
            return False, None

        for file_name in image_files:
            known_face_path = os.path.join(self.database_path, file_name)
            known_face = cv2.imread(known_face_path)
            if known_face is None:
                continue

            try:
                result = DeepFace.verify(detected_face_rgb, known_face, model_name=model_name)
                if result.get("verified"):
                    print(f"Match found with {file_name}.")
                    return True, file_name
            except Exception as e:
                print(f"Error verifying with {file_name}: {e}")
                
        return False, None
    
    def verify_face_in_frame(self, known_face_image_path, frame, model_name="Facenet"):
        """
        Verifies if the face in the provided frame matches the known face image.

        Parameters:
        - known_face_image_path: Path to the image in the database for verification.
        - frame: The video frame containing the detected face to verify.
        - model_name: DeepFace model name to use for verification.

        Returns:
        - True if a match is found, False otherwise.
        """
        try:
            # Read the known face image from the database
            known_face = cv2.imread(known_face_image_path)
            if known_face is None:
                print("Error: Unable to read the known face image.")
                return False

            # Extract faces from the frame using DeepFace
            faces = DeepFace.extract_faces(img_path=frame, detector_backend="mtcnn")
            if len(faces) == 0:
                print("No faces detected in the frame.")
                return False

            # Assume the first face in the frame is the detected face
            detected_face = faces[0]["face"]

            # Convert to RGB if needed
            detected_face_rgb = cv2.cvtColor(np.uint8(detected_face * 255), cv2.COLOR_BGR2RGB)

            # Verify the detected face against the known face
            result = DeepFace.verify(detected_face_rgb, known_face, model_name=model_name)

            # Check if faces match
            if result.get("verified"):
                print("Face verified as a match!")
                return True
            else:
                print("Face does not match.")
                return False

        except Exception as e:
            print(f"Error verifying face in frame: {e}")
            return False

    def save_detected_face_to_database(self, detected_face_rgb):
        """
        Save a new detected face to the database.

        Parameters:
        - detected_face_rgb: RGB image of the detected face (preprocessed).
        """
        file_name = f"face_{self.face_id}.jpg"
        file_path = os.path.join(self.database_path, file_name)
        cv2.imwrite(file_path, cv2.cvtColor(detected_face_rgb, cv2.COLOR_RGB2BGR))
        print(f"New face saved to database as {file_path}")
        self.face_id += 1

    def object_detection(self, frame):
        """
        Process a single frame for face detection and recognition.

        Parameters:
        - frame: The video frame to process.

        Returns:
        - Processed frame (optional for visualization purposes).
        """
        try:
            # Step 1: Extract faces
            faces = DeepFace.extract_faces(img_path=frame, detector_backend="mtcnn")
            
            if not faces:
                print("No faces detected in the frame.")
                return frame
        
            for face_data in faces:
                detected_face = face_data["face"]
                detected_face_resized = np.uint8(detected_face * 255)  # Ensure uint8 format
                detected_face_rgb = cv2.cvtColor(detected_face_resized, cv2.COLOR_BGR2RGB)
                
                # Verify face against database
                face_found, matched_file_name = self.verify_face_with_database(detected_face_rgb)

                if not face_found:
                    print("Unknown face detected!")
                    while True:
                        user_input = input("Do you want to store the detected face in the database? (yes/no): ").strip().lower()
                    
                        if user_input == "yes":
                            self.save_detected_face_to_database(detected_face_rgb)
                            print("Face stored successfully.")
                            break
                        elif user_input == "no":
                            print("Face not stored. Returning to normal frame.")
                            self.active_task = "normal"
                            return frame  # Return the normal frame if user says no
                        else:
                            print("Invalid input. Please enter 'yes' or 'no'.")

        except Exception as e:
            print(f"Error processing frame: {e}")

        return frame
    
    def emotion_analysis(self, frame):
        # Example emotion analysis logic
        result = DeepFace.analyze(frame, actions=["emotion"])
        # print("Emotion detected:", result[0]["dominant_emotion"])
        return result[0]["dominant_emotion"]
    

    def process_frame(self, frame):
        """
        Process the frame based on the active task.
        """
        if self.active_task == "face_detection":
            is_verified = self.verify_face_in_frame("db/detected_face_2.jpg", frame)
            if is_verified:
                print("Hello, Mr. Singh")
                convert_and_play_audio("output.mp3","output.wav")
                
            return frame

        elif self.active_task == "emotion_analysis":
            emotion_result = self.emotion_analysis(frame)
            print(emotion_result)
            return frame

        elif self.active_task == "object_detection":
            
            return self.object_detection(frame)

        elif self.active_task == "normal":
            return frame
        
        else:
            return frame  # No active task, return the original frame
        
    def check_task_switch(self):
        """
        Dynamically switch tasks based on keypress.
        """
        key_map = {
            "f": "face_detection",
            "e": "emotion_analysis",
            "o": "object_detection",
            "a": "analyze",
            "n": "normal",
        }

        while self.running:
            for key, task in key_map.items():
                if keyboard.is_pressed(key):
                    if self.active_task != task:
                        print(f"Switching to {task.title().replace('_', ' ')}...")
                        self.active_task = task
                    return

            if keyboard.is_pressed("q"):
                self.running = False
                print("Exiting application.")
                return 


    def ask_query_about_frame(self,frame, query):
        """
        Upload the frame to GenAI and ask a query about it.

        Parameters:
        - frame: The image frame from the video capture.
        - query: The query to ask about the frame.

        Returns:
        - Response from the GenAI model.
        """
        try:
            # Save the frame as a temporary image
            temp_file = "temp_frame.jpg"
            cv2.imwrite(temp_file, frame)

            # Upload the image to GenAI
            myfile = genai.upload_file(temp_file)

            # Initialize the model and send the query
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content(
                [myfile, "\n\n", query]
            )

            # Return the result
            return result.text
        
        except Exception as e:
            return f"Error processing query: {e}"
    
    def analyze_frame(self, frame):
        """
        Function to trigger GenAI API analysis for the current frame.
        """
        # query = "What do you see in this photo?"  # Query for analyzing the current frame
        user_input = input("query : ").strip().lower()
        print("Processing the current frame for analysis...")

        # Call the GenAI API to analyze the frame
        response = self.ask_query_about_frame(frame, user_input)
        audio_file_path = text_to_speech(response, "sk_632eb96bd5ff0ac2d1695753c761f837209eba3287b3c6a4", "cgSgspJ2msm6clMCkdW9")
        convert_and_play_audio(audio_file_path)
        print("Query Result:", response)

    
    def run(self):
        """
        Main loop to capture and process video frames.
        """
        cap = cv2.VideoCapture(0)
        
        # self.active_task = "normal"
        print("Press 'f' for Face Detection, 'e' for Emotion Analysis, 'o' for Object Detection,'a' to analyze the current frame, 'q' to Quit.")

        prev_switch_time = 0  # To debounce keypresses
        debounce_interval = 0.2  # Seconds

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break
            
            

            # Task switching with debounce
            current_time = time.time()
            if current_time - prev_switch_time > debounce_interval:
                self.check_task_switch()
                prev_switch_time = current_time
                
            # Display the processed frame

            # Process frame for the current active task
            processed_frame = self.process_frame(frame)
            # cv2.imshow("Vision Assistant", frame)
            
            # If the task is 'analyze', process the frame for query analysis
            if self.active_task == "analyze" and not self.query_triggered:
                self.analyze_frame(frame)
                self.query_triggered = True
                time.sleep(2)  # Add delay to avoid triggering multiple queries

            # Reset query trigger after some time
            if self.query_triggered:
                self.query_frame_count += 1
                if self.query_frame_count > 30:  # Reset trigger after ~1 second
                    self.query_triggered = False
                    self.query_frame_count = 0

            # Display the processed frame
            cv2.imshow("Vision Pipeline", processed_frame)
            
            # Check for keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit the application
                print("Exiting...")
                break
            
        cap.release()
        cv2.destroyAllWindows()

# Initialize and run the pipeline
pipeline = VisionPipeline()
pipeline.run()


