import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/cloudwalkoffice/face_detection/face_detection_env/lib/python3.11/site-packages/cv2/qt/plugins"

import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pickle
import threading

class FaceDetector:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.saved_faces_dir = "saved_faces"
        self.encodings_file = "known_faces.pkl"
        self.registration_dir = "new_registrations"
        self.logs_dir = "detection_logs"  # New directory for daily logs
        self.face_trackers = {}  # Track faces between frames
        self.processed_unknowns = set()  # Keep track of recently processed unknown faces
        self.cap = None
        self.log_callback = None
        self.last_detection_times = {}  # Track last detection time for each person
        
        # Create necessary directories
        for directory in [self.saved_faces_dir, self.registration_dir, self.logs_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
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

    def generate_hashcode(self):
        """Generate a unique hashcode for unknown faces"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"Person_{timestamp}"

    def get_log_file_path(self, date=None):
        """Get the path for the log file of a specific date"""
        if date is None:
            date = datetime.now()
        return os.path.join(self.logs_dir, f"detections_{date.strftime('%Y%m%d')}.txt")

    def log_detection(self, name, timestamp):
        """Log a detection to the daily log file"""
        log_file = self.get_log_file_path(datetime.strptime(timestamp.split()[0], "%Y-%m-%d"))
        print(f"{name} detected at {timestamp}")  # Console output
        
        with open(log_file, "a", buffering=1) as f:  # Line buffering
            f.write(f"{name} was detected at the office at {timestamp}\n")
            f.flush()  # Force write to disk

    def update_logs_with_new_name(self, old_name, new_name):
        """Update all log files with the new name"""
        try:
            # Get all log files
            log_files = [f for f in os.listdir(self.logs_dir) if f.startswith("detections_")]
            updates_made = False
            
            for log_file in log_files:
                file_path = os.path.join(self.logs_dir, log_file)
                temp_file_path = file_path + ".tmp"
                
                with open(file_path, 'r') as f_in, open(temp_file_path, 'w') as f_out:
                    for line in f_in:
                        # Replace old name with new name
                        if old_name in line:
                            new_line = line.replace(old_name, new_name)
                            f_out.write(new_line)
                            updates_made = True
                        else:
                            f_out.write(line)
                
                # Replace original file with updated file
                os.replace(temp_file_path, file_path)
            
            if updates_made:
                print(f"Updated log files: replaced '{old_name}' with '{new_name}'")
            return True
        except Exception as e:
            print(f"Error updating log files: {str(e)}")
            return False

    def update_face_name(self, old_name, new_name):
        """Update the name of an existing face"""
        try:
            if old_name in self.known_names:
                # Get index of the face to update
                idx = self.known_names.index(old_name)
                
                # Update name in memory
                self.known_names[idx] = new_name
                
                # Update the name in saved files
                old_files = [f for f in os.listdir(self.saved_faces_dir) if f.startswith(old_name + "_")]
                for old_file in old_files:
                    # Create new filename preserving the timestamp
                    timestamp = old_file.replace(old_name + "_", "", 1)
                    new_file = f"{new_name}_{timestamp}"
                    old_path = os.path.join(self.saved_faces_dir, old_file)
                    new_path = os.path.join(self.saved_faces_dir, new_file)
                    os.rename(old_path, new_path)
                    print(f"Renamed file: {old_file} -> {new_file}")
                
                # Update log files with new name
                self.update_logs_with_new_name(old_name, new_name)
                
                # Save updated known faces
                self.save_known_faces()
                print(f"Updated name from {old_name} to {new_name}")
                return True
            return False
        except Exception as e:
            print(f"Error updating face name: {str(e)}")
            return False

    def save_face(self, frame, face_location, name=None, auto_capture=False):
        try:
            # Extract face from frame
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            # Get face encoding first to check for existing face
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encoding = self.get_face_encoding(rgb_frame, face_location)
            
            if face_encoding is not None:
                face_encoding = np.array(face_encoding)
                
                # Check if this face is already known
                if self.known_faces:
                    distances = face_recognition.face_distance(self.known_faces, face_encoding)
                    min_distance_idx = np.argmin(distances)
                    min_distance = distances[min_distance_idx]
                    
                    if min_distance < 0.4:  # Face already exists
                        existing_name = self.known_names[min_distance_idx]
                        if not auto_capture and name is not None and existing_name != name:
                            # Update name if it's a manual save with a different name
                            return self.update_face_name(existing_name, name)
                        print(f"This face is already registered as {existing_name}")
                        return False
                
                # If we get here, it's a new face or a manual save with a new name
                if name is None and not auto_capture:
                    name = input("Enter name for this face: ")
                elif name is None and auto_capture:
                    name = self.generate_hashcode()
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(self.saved_faces_dir, filename)
                
                # Save the face image
                cv2.imwrite(filepath, face_image)
                
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

    def delete_face(self, name):
        """Delete a face from the database and remove its files"""
        try:
            if name in self.known_names:
                # Get index of the face to delete
                idx = self.known_names.index(name)
                
                # Remove from memory
                self.known_faces.pop(idx)
                self.known_names.remove(name)
                
                # Remove associated files
                files_to_remove = [f for f in os.listdir(self.saved_faces_dir) if f.startswith(name)]
                for file in files_to_remove:
                    file_path = os.path.join(self.saved_faces_dir, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file}")
                    except Exception as e:
                        print(f"Error deleting file {file}: {str(e)}")
                
                # Save updated known faces
                self.save_known_faces()
                print(f"Deleted face: {name}")
                return True
            else:
                print(f"Face not found: {name}")
                return False
        except Exception as e:
            print(f"Error deleting face: {str(e)}")
            return False

    def should_log_detection(self, name):
        """Check if enough time has passed to log this detection"""
        current_time = datetime.now()
        if name in self.last_detection_times:
            time_diff = (current_time - self.last_detection_times[name]).total_seconds()
            if time_diff < 5:  # 5 second cooldown
                return False
        self.last_detection_times[name] = current_time
        return True

    def get_frame_with_detections(self):
        """Get a frame from the camera with face detections drawn on it"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False, None
            
            # Lower resolution for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Keep 30fps for smooth video

        success, frame = self.cap.read()
        if not success:
            return False, None

        height, width = frame.shape[:2]
        square_size = min(width, height) // 2
        x1 = (width - square_size) // 2
        y1 = (height - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces using HOG model for better performance
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for face_location, face_encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = face_location
                face_center_x = (left + right) // 2
                face_center_y = (top + bottom) // 2
                
                name = "Unknown"
                if self.known_faces:
                    distances = face_recognition.face_distance(self.known_faces, face_encoding)
                    if len(distances) > 0:
                        best_match_index = np.argmin(distances)
                        if distances[best_match_index] < 0.6:
                            name = self.known_names[best_match_index]
                
                is_inside = (x1 < face_center_x < x2 and y1 < face_center_y < y2)
                color = (0, 0, 255) if is_inside else (255, 0, 0)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Handle unknown faces in green box
                if is_inside and name == "Unknown":
                    face_key = f"{top}_{right}_{bottom}_{left}"
                    if face_key not in self.processed_unknowns:
                        self.processed_unknowns.add(face_key)
                        face_frame = frame[top:bottom, left:right].copy()
                        threading.Thread(
                            target=self.save_face,
                            args=(frame, face_location, None, True),
                            daemon=True
                        ).start()
                
                # Log detections for known faces with cooldown
                elif is_inside and name != "Unknown" and self.should_log_detection(name):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(self, 'log_callback'):
                        self.log_callback(name, timestamp)
                    else:
                        self.log_detection(name, timestamp)
                        
        except Exception as e:
            print(f"Error in detection: {str(e)}")

        return True, frame

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Camera resources released")

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run_detection() 