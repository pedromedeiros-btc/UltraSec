import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/cloudwalkoffice/face_detection/face_detection_env/lib/python3.11/site-packages/cv2/qt/plugins"

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
        self.registration_dir = "new_registrations"
        self.logs_dir = "detection_logs"  # New directory for daily logs
        self.face_trackers = {}  # Track faces between frames
        self.processed_unknowns = set()  # Keep track of recently processed unknown faces
        
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

    def show_faces_to_delete(self):
        """Show list of faces that can be deleted and handle deletion"""
        if not self.known_names:
            print("No faces in database to delete.")
            return False
            
        print("\nAvailable faces to delete:")
        print("---------------------------")
        for i, name in enumerate(self.known_names, 1):
            print(f"{i}. {name}")
        print("---------------------------")
        
        while True:
            choice = input("Enter the name or number to delete (or 'cancel' to abort): ")
            if choice.lower() == 'cancel':
                print("Deletion cancelled.")
                return False
            
            # Try to get the name to delete
            name_to_delete = None
            
            # Check if input is a number
            try:
                index = int(choice)
                if 1 <= index <= len(self.known_names):
                    name_to_delete = self.known_names[index - 1]
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(self.known_names)}")
                    continue
            except ValueError:
                # Input is not a number, try to match name directly
                if choice in self.known_names:
                    name_to_delete = choice
                else:
                    print(f"'{choice}' not found. Please enter a valid name or number from the list.")
                    continue
            
            # Confirm deletion
            print(f"\nYou are about to delete '{name_to_delete}'")
            confirmation = input("Type 'confirm' to proceed with deletion: ")
            
            if confirmation.lower() == 'confirm':
                if self.delete_face(name_to_delete):
                    print(f"Successfully deleted {name_to_delete}")
                    return True
                else:
                    print(f"Failed to delete {name_to_delete}")
                    return False
            else:
                print("Deletion cancelled.")
                return False

    def show_registered_faces(self):
        """Show list of all registered faces with option to edit names"""
        if not self.known_names:
            print("\nNo faces registered in the database.")
            return
            
        while True:
            print("\nRegistered faces:")
            print("----------------")
            for i, name in enumerate(self.known_names, 1):
                print(f"{i}. {name}")
            print("----------------")
            print("Type 'edit' to rename a face, or press Enter to continue")
            
            choice = input("> ").lower()
            if not choice:  # Empty input (just Enter)
                break
            elif choice == 'edit':
                self.edit_face_name()

    def edit_face_name(self):
        """Handle the face name editing process"""
        print("\nSelect face to rename:")
        print("---------------------------")
        for i, name in enumerate(self.known_names, 1):
            print(f"{i}. {name}")
        print("---------------------------")
        
        while True:
            choice = input("Enter the name or number to edit (or 'cancel' to abort): ")
            if choice.lower() == 'cancel':
                print("Edit cancelled.")
                return False
            
            # Try to get the name to edit
            old_name = None
            
            # Check if input is a number
            try:
                index = int(choice)
                if 1 <= index <= len(self.known_names):
                    old_name = self.known_names[index - 1]
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(self.known_names)}")
                    continue
            except ValueError:
                # Input is not a number, try to match name directly
                if choice in self.known_names:
                    old_name = choice
                else:
                    print(f"'{choice}' not found. Please enter a valid name or number from the list.")
                    continue
            
            # Get new name
            new_name = input(f"Enter new name for '{old_name}': ").strip()
            if not new_name:
                print("Name cannot be empty. Edit cancelled.")
                return False
                
            # Confirm rename
            print(f"\nYou are about to rename '{old_name}' to '{new_name}'")
            confirmation = input("Type 'confirm' to proceed: ")
            
            if confirmation.lower() == 'confirm':
                if self.update_face_name(old_name, new_name):
                    print(f"Successfully renamed {old_name} to {new_name}")
                    return True
                else:
                    print(f"Failed to rename {old_name}")
                    return False
            else:
                print("Edit cancelled.")
                return False

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        process_every_n_frames = 4
        frame_count = 0
        
        process_this_frame = True
        last_save_time = {}  # Track last save time for each face location
        last_detection_time = {}  # Track last detection time for each name
        current_face_in_square = None  # Track the current face name in green square
        
        # Create debug window
        debug_window_name = 'Debug Info'
        debug_height = 300
        debug_width = 400
        debug_image = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
        cv2.namedWindow(debug_window_name)

        print("Known faces at startup:", self.known_names)
        print("Controls: 'S' to save face, 'D' to delete a face, 'list' to show registered faces, 'Q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % process_every_n_frames != 0:
                continue

            current_time = datetime.now()
            current_face_in_square = None  # Reset at each frame
            
            # Calculate square coordinates FIRST
            height, width = frame.shape[:2]
            square_size = min(width, height) // 2
            x1 = (width - square_size) // 2
            y1 = (height - square_size) // 2
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            if process_this_frame:
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Detect faces using HOG model for better performance
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # Clear debug window
                    debug_image.fill(0)
                    
                    face_names = []
                    
                    for i, face_encoding in enumerate(face_encodings):
                        name = "Unknown"
                        face_location = face_locations[i]
                        top, right, bottom, left = face_location
                        face_center_x = (left + right) // 2
                        face_center_y = (top + bottom) // 2
                        
                        if self.known_faces:
                            # Compare faces
                            distances = face_recognition.face_distance(self.known_faces, face_encoding)
                            distances = np.array(distances)
                            
                            if len(distances) > 0:
                                best_match_index = np.argmin(distances)
                                
                                # Explicit boolean conversion
                                if bool(distances[best_match_index] < 0.6):
                                    name = self.known_names[best_match_index]
                                    
                                    # Only log if face is inside green square and enough time has passed
                                    if (x1 < face_center_x < x2 and y1 < face_center_y < y2):
                                        current_face_in_square = name  # Track face in square
                                        if (name not in last_detection_time or 
                                            (current_time - last_detection_time[name]).total_seconds() > 5):
                                            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                            self.log_detection(name, timestamp)
                                            last_detection_time[name] = current_time
                        
                        # Auto-capture unknown faces in green square
                        if name == "Unknown" and (x1 < face_center_x < x2 and y1 < face_center_y < y2):
                            current_face_in_square = "Unknown"  # Track unknown face in square
                            # Check if we haven't recently saved this face
                            face_key = f"{top}_{right}_{bottom}_{left}"
                            if (face_key not in last_save_time or 
                                (current_time - last_save_time[face_key]).total_seconds() > 5):  # 5 second cooldown
                                
                                if self.save_face(frame, face_location, auto_capture=True):
                                    last_save_time[face_key] = current_time
                                    # Update name with the new hashcode
                                    name = self.known_names[-1]
                                    current_face_in_square = name  # Update tracked face
                        
                        face_names.append(name)
                        
                        # Show distances in debug window
                        if self.known_faces:
                            for j, (dist, known_name) in enumerate(zip(distances, self.known_names)):
                                debug_text = f"{known_name}: {dist:.3f}"
                                cv2.putText(debug_image, debug_text, (10, 30 + j*30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Show debug window
                    cv2.imshow(debug_window_name, debug_image)
                    
                except Exception as e:
                    print(f"Error in detection: {str(e)}")
                    face_locations = []
                    face_names = []

            process_this_frame = not process_this_frame

            # Draw green square in the center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw face rectangles and names
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                face_center_x = (left + right) // 2
                face_center_y = (top + bottom) // 2
                
                # Change color if face is in green square
                if (x1 < face_center_x < x2 and y1 < face_center_y < y2):
                    color = (0, 0, 255)  # Red for faces in green square
                    if name == "Unknown":
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
            elif key & 0xFF == ord('d'):
                # Show list of faces and handle deletion
                self.show_faces_to_delete()
            elif key & 0xFF == ord('l'):  # 'l' for list
                # Show registered faces
                self.show_registered_faces()

            # Check for new registrations
            for file in os.listdir(self.registration_dir):
                if file.endswith('.jpg') or file.endswith('.png'):
                    name, _ = os.path.splitext(file)
                    image_path = os.path.join(self.registration_dir, file)
                    if self.register_face_from_file(image_path, name):
                        os.remove(image_path)  # Remove after successful registration

        cap.release()
        cv2.destroyAllWindows()

    def save_remote_face(self, image_path, name):
        frame = cv2.imread(image_path)
        if frame is None:
            return False
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 1:
            return self.save_face(frame, face_locations[0], name)
        return False

    def register_face_from_file(self, image_path, name):
        try:
            # Load and process the image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not load image: {image_path}")
                return False
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) == 1:
                return self.save_face(frame, face_locations[0], name)
            else:
                print(f"Found {len(face_locations)} faces, need exactly 1 face")
                return False
            
        except Exception as e:
            print(f"Error registering face: {str(e)}")
            return False

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run_detection() 