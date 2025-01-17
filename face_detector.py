import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pickle

class FaceDetector:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.saved_faces_dir = "saved_faces"
        self.encodings_file = "known_faces.pkl"
        
        # Create directory for saved faces if it doesn't exist
        if not os.path.exists(self.saved_faces_dir):
            os.makedirs(self.saved_faces_dir)
            
        # Load known faces if file exists
        self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data['encodings']
                    self.known_names = data['names']
                    print(f"Loaded {len(self.known_faces)} known faces: {self.known_names}")
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_faces = []
                self.known_names = []

    def save_known_faces(self):
        with open(self.encodings_file, 'wb') as f:
            data = {
                'encodings': self.known_faces,
                'names': self.known_names
            }
            pickle.dump(data, f)

    def get_face_encoding(self, rgb_image, face_location):
        try:
            # Use face_recognition's built-in function for the whole image
            encodings = face_recognition.face_encodings(rgb_image, [face_location])
            if len(encodings) > 0:
                return encodings[0]
            print("No face encoding found")
            return None
        except Exception as e:
            print(f"Error in get_face_encoding: {str(e)}")
            return None

    def save_face(self, frame, face_location, name=None):
        try:
            # Extract face from frame
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            if name is None:
                name = input("Enter name for this face: ")
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(self.saved_faces_dir, filename)
            
            # Save the face image
            cv2.imwrite(filepath, face_image)
            
            # Get face encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encoding = self.get_face_encoding(rgb_frame, face_location)
            
            if face_encoding is not None:
                self.known_faces.append(face_encoding)
                self.known_names.append(name)
                self.save_known_faces()
                print(f"Face saved as {filename}")
                print(f"Known faces: {self.known_names}")
                return True
            return False
            
        except Exception as e:
            print(f"Error saving face: {str(e)}")
            return False

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        process_this_frame = True
        
        # Create debug window
        debug_window_name = 'Debug Info'
        debug_height = 300
        debug_width = 400
        debug_image = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
        cv2.namedWindow(debug_window_name)

        print("Known faces at startup:", self.known_names)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if process_this_frame:
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # Clear debug window
                    debug_image.fill(0)
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        name = "Unknown"
                        
                        if len(self.known_faces) > 0:
                            # Compare faces
                            distances = face_recognition.face_distance(self.known_faces, face_encoding)
                            if len(distances) > 0:
                                best_match_index = np.argmin(distances)
                                if distances[best_match_index] < 0.6:
                                    name = self.known_names[best_match_index]
                                
                                # Show distances in debug window
                                for i, (dist, known_name) in enumerate(zip(distances, self.known_names)):
                                    debug_text = f"{known_name}: {dist:.3f}"
                                    cv2.putText(debug_image, debug_text, (10, 30 + i*30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                    print(f"Distance to {known_name}: {dist:.3f}")
                        
                        face_names.append(name)
                    
                    # Show debug window
                    cv2.imshow(debug_window_name, debug_image)
                    
                except Exception as e:
                    print(f"Error in detection: {str(e)}")
                    face_locations = []
                    face_names = []

            process_this_frame = not process_this_frame

            # Draw green square in the center
            height, width = frame.shape[:2]
            square_size = min(width, height) // 2
            x1 = (width - square_size) // 2
            y1 = (height - square_size) // 2
            x2 = x1 + square_size
            y2 = y1 + square_size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw face rectangles and names
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                face_center_x = (left + right) // 2
                face_center_y = (top + bottom) // 2
                
                # Change color if face is in green square
                if (x1 < face_center_x < x2 and y1 < face_center_y < y2):
                    color = (0, 0, 255)  # Red for faces in green square
                    cv2.putText(frame, "Press 'S' to save face", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    color = (255, 0, 0)  # Blue for faces outside

                # Draw box around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Show frame
            cv2.imshow('Face Recognition', frame)

            # Handle keyboard input
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                # Save faces that are inside the green square
                for face_location, name in zip(face_locations, face_names):
                    top, right, bottom, left = face_location
                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2
                    
                    if (x1 < face_center_x < x2 and y1 < face_center_y < y2):
                        self.save_face(frame, face_location)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run_detection() 