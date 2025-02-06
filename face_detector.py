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
import subprocess
from PIL import Image
import io
import requests
from requests.auth import HTTPDigestAuth
import time

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
        self.last_processed_clear = datetime.now()  # Add this line
        self.cap = None  # Changed from camera_process to cap
        self.log_callback = None
        self.last_detection_times = {}  # Track last detection time for each person
        self.door_controller_url = "http://100.94.146.43:8081/door"  # Pi's IP address
        self.door_auth = HTTPDigestAuth('admin', '9cby@GmP')
        self.last_door_action_time = 0
        self.door_cooldown = 0.5  # 500ms cooldown between door actions
        
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
            print(f"Attempting to save face. Name: {name}, Auto capture: {auto_capture}")
            
            # Extract face from frame with larger padding (40% on each side)
            top, right, bottom, left = face_location
            height, width = frame.shape[:2]
            
            # Calculate padding (40% of face size)
            pad_x = int((right - left) * 0.4)
            pad_y = int((bottom - top) * 0.4)
            
            # Ensure padded coordinates are within frame bounds
            left_pad = max(0, left - pad_x)
            right_pad = min(width, right + pad_x)
            top_pad = max(0, top - pad_y)
            bottom_pad = min(height, bottom + pad_y)
            
            # Extract padded face region from clean frame
            face_image = frame[top_pad:bottom_pad, left_pad:right_pad].copy()
            
            # Get face encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encoding = self.get_face_encoding(rgb_frame, face_location)
            
            if face_encoding is not None:
                face_encoding = np.array(face_encoding)
                print("Got face encoding successfully")
                
                # Check if this face is already known
                if self.known_faces:
                    distances = face_recognition.face_distance(self.known_faces, face_encoding)
                    min_distance_idx = np.argmin(distances)
                    min_distance = distances[min_distance_idx]
                    print(f"Minimum distance to known faces: {min_distance:.3f}")
                    
                    if min_distance < 0.6:
                        existing_name = self.known_names[min_distance_idx]
                        print(f"Face matches existing person: {existing_name}")
                        if not auto_capture and name is not None and existing_name != name:
                            return self.update_face_name(existing_name, name)
                        return False
                
                # Generate name for unknown face
                if name is None:
                    if auto_capture:
                        name = self.generate_hashcode()
                        print(f"Generated new name for unknown face: {name}")
                    else:
                        name = input("Enter name for this face: ")
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(self.saved_faces_dir, filename)
                
                # Resize the face image to be larger (800px height while maintaining aspect ratio)
                aspect_ratio = face_image.shape[1] / face_image.shape[0]
                target_height = 800
                target_width = int(target_height * aspect_ratio)
                face_image = cv2.resize(face_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Save the clean, resized face image
                cv2.imwrite(filepath, face_image)
                print(f"Saved face image to: {filepath}")
                
                self.known_faces.append(face_encoding)
                self.known_names.append(name)
                self.save_known_faces()
                print(f"Added new face to known faces: {name}")

                # Notify web app about new face
                if hasattr(self, 'socketio'):
                    print("Notifying web app of new face")
                    self.socketio.emit('faces_updated')
                
                return True
                
            print("Failed to get face encoding")
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

    def start_camera(self):
        """Initialize camera capture"""
        try:
            if self.cap is not None:
                self.cleanup()
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Failed to open camera")
                return False
                
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("Started camera capture with optimized settings")
            return True
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False

    def control_door(self, action):
        """Control the door with rate limiting"""
        try:
            current_time = time.time()
            if current_time - self.last_door_action_time < self.door_cooldown:
                print("Door action too soon, waiting for cooldown")
                return False

            print(f"Attempting to {action} door...")
            params = {"action": action, "channel": "1"}
            
            # Print full URL and auth details
            full_url = f"{self.door_controller_url}?action={action}&channel=1"
            print(f"Sending request to: {full_url}")
            print(f"Using digest auth with username: {self.door_auth.username}")
            
            # Create a session with specific interface binding
            session = requests.Session()
            session.mount("http://", requests.adapters.HTTPAdapter(max_retries=1))
            
            # Add more timeout options and debugging
            response = session.get(
                self.door_controller_url,
                params=params,
                auth=self.door_auth,
                timeout=5,
                verify=False,  # Skip SSL verification if needed
                headers={
                    'Connection': 'close',
                    'Host': '192.168.68.210'
                }
            )
            
            self.last_door_action_time = current_time
            
            # Print full response details
            print(f"Door response status code: {response.status_code}")
            print(f"Door response text: {response.text}")
            print(f"Door response headers: {dict(response.headers)}")
            
            if response.text.strip() == "OK":
                print(f"Door {action} successful")
                return True
            else:
                print(f"Door {action} failed: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("Door controller request timed out - check if device is accessible")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"Failed to connect to door controller: {str(e)}")
            print("Check network connectivity and routes")
            return False
        except Exception as e:
            print(f"Error controlling door: {str(e)}")
            print(f"Full error details: {repr(e)}")
            return False

    def open_door(self):
        """Open the door"""
        return self.control_door("openDoor")

    def close_door(self):
        """Close the door"""
        return self.control_door("closeDoor")

    def should_open_door(self, name):
        """Check if the person is authorized to open the door"""
        # Add debugging
        is_authorized = not name.startswith("Person_")
        print(f"Checking authorization for {name}: {'authorized' if is_authorized else 'not authorized'}")
        return is_authorized

    def get_frame_with_detections(self):
        """Get a frame from camera with face detections"""
        if self.cap is None or not self.cap.isOpened():
            if not self.start_camera():
                return False, None

        try:
            ret, original_frame = self.cap.read()
            if not ret or original_frame is None:
                print("Failed to read frame")
                return False, None

            # Make a copy for drawing
            display_frame = original_frame.copy()
            height, width = display_frame.shape[:2]

            # Draw detection box
            square_size = min(width, height) * 2 // 3
            x1 = (width - square_size) // 2
            y1 = (height - square_size) // 2
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Process faces
            rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model="small")
                
                for face_location, face_encoding in zip(face_locations, face_encodings):
                    top, right, bottom, left = face_location
                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2
                    
                    name = "Unknown"
                    if self.known_faces:
                        distances = face_recognition.face_distance(np.array(self.known_faces), face_encoding)
                        if len(distances) > 0:
                            best_match_index = np.argmin(distances)
                            min_distance = distances[best_match_index]
                            if min_distance < 0.6:
                                name = self.known_names[best_match_index]
                    
                    is_inside = (x1 < face_center_x < x2 and y1 < face_center_y < y2)
                    color = (0, 0, 255) if is_inside else (255, 0, 0)
                    
                    # Draw rectangles and text on display frame only
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display_frame, name, (left + 6, bottom - 6),
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Only process faces that are inside the green box
                    if is_inside:
                        if name == "Unknown":
                            print("Found unknown face in green box")
                            self.save_face(original_frame, face_location, None, True)
                            if hasattr(self, 'socketio'):
                                self.socketio.emit('faces_updated')
                                print("Notified web app of new face")
                        else:
                            # Add more debugging for door control flow
                            print(f"Known face detected in green box: {name}")
                            if self.should_open_door(name):
                                print(f"Attempting to open door for authorized person: {name}")
                                try:
                                    if self.open_door():
                                        print(f"Successfully opened door for {name}")
                                        print("Starting timer to close door...")
                                        threading.Timer(5.0, self.close_door).start()
                                    else:
                                        print(f"Failed to open door for {name}")
                                except Exception as e:
                                    print(f"Error in door control sequence: {str(e)}")
                            else:
                                print(f"Person not authorized to open door: {name}")
                            
                            # Log the detection as before
                            if self.should_log_detection(name):
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                if hasattr(self, 'log_callback'):
                                    self.log_callback(name, timestamp)
                                else:
                                    self.log_detection(name, timestamp)
            
            return True, display_frame

        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return False, None

    def cleanup(self):
        """Clean up camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera resources released")

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run_detection() 