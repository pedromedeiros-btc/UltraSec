from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import threading
import json
import os
from datetime import datetime
from face_detector import FaceDetector
import base64
import numpy as np
import subprocess
import signal
import sys
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detector = FaceDetector()
camera_thread = None
camera_running = False
frame_buffer = None
ngrok_process = None

def start_ngrok():
    """Start ngrok in the background"""
    global ngrok_process
    try:
        ngrok_process = subprocess.Popen(
            ['ngrok', 'http', '5000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Started ngrok process")
    except Exception as e:
        print(f"Error starting ngrok: {str(e)}")

def cleanup():
    """Cleanup function to handle graceful shutdown"""
    global camera_running, ngrok_process
    print("Cleaning up...")
    camera_running = False
    if ngrok_process:
        ngrok_process.terminate()
    if detector:
        detector.cleanup()  # Clean up camera resources

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def read_latest_logs():
    """Read the latest logs from today's log file"""
    try:
        today = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(detector.logs_dir, f"detections_{today}.txt")
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-50:]]  # Get last 50 lines
        return []
    except Exception as e:
        print(f"Error reading logs: {str(e)}")
        return []

def log_callback(name, timestamp):
    """Callback function for face detection logs"""
    log_message = f"{name} was detected at the office at {timestamp}"
    
    # Write to log file
    today = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(detector.logs_dir, f"detections_{today}.txt")
    
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')
    except Exception as e:
        print(f"Error writing to log file: {str(e)}")
    
    # Emit to all connected clients
    try:
        socketio.emit('new_log', {'message': log_message}, broadcast=True)
        print(f"Emitted log: {log_message}")  # Debug print
    except Exception as e:
        print(f"Error emitting log: {str(e)}")

# Set the log callback
detector.log_callback = log_callback

def camera_feed():
    global frame_buffer, camera_running
    camera_running = True
    print("Starting camera feed...")
    
    try:
        while camera_running:
            success, frame = detector.get_frame_with_detections()
            if success:
                # Convert frame to base64 for web streaming
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Emit frame via WebSocket
                socketio.emit('frame_update', {'frame': frame_base64})
                
                # Store latest frame in buffer
                frame_buffer = frame
                
                # Small delay to prevent freezing
                time.sleep(0.01)  # Reduced delay for better responsiveness
            else:
                print("Failed to get frame from camera")
                time.sleep(1)  # Wait before retrying
    except Exception as e:
        print(f"Error in camera feed: {str(e)}")
    finally:
        print("Camera feed stopped")

@app.route('/')
def index():
    logs = read_latest_logs()
    return render_template('index.html', initial_logs=logs)

@app.route('/api/faces', methods=['GET'])
def get_faces():
    return jsonify({
        'faces': [{'name': name, 'id': idx} for idx, name in enumerate(detector.known_names)]
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        logs = read_latest_logs()
        return jsonify({'logs': logs})
    except Exception as e:
        print(f"Error getting logs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/faces', methods=['POST'])
def add_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    name = request.form.get('name', 'Unknown')
    
    # Save image temporarily
    temp_path = f'temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
    image_file.save(temp_path)
    
    # Process image with face detector
    frame = cv2.imread(temp_path)
    if frame is not None:
        success = detector.save_face(frame, None, name=name)
        os.remove(temp_path)
        
        if success:
            socketio.emit('faces_updated')
            return jsonify({'message': 'Face added successfully'})
    
    return jsonify({'error': 'Failed to add face'}), 400

@app.route('/api/faces/<int:face_id>', methods=['PUT'])
def update_face(face_id):
    if face_id >= len(detector.known_names):
        return jsonify({'error': 'Face not found'}), 404
    
    data = request.get_json()
    new_name = data.get('name')
    
    if new_name:
        old_name = detector.known_names[face_id]
        if detector.update_face_name(old_name, new_name):
            socketio.emit('faces_updated')
            return jsonify({'message': 'Face updated successfully'})
    
    return jsonify({'error': 'Failed to update face'}), 400

@app.route('/api/faces/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    if face_id >= len(detector.known_names):
        return jsonify({'error': 'Face not found'}), 404
    
    name = detector.known_names[face_id]
    if detector.delete_face(name):
        socketio.emit('faces_updated')
        return jsonify({'message': 'Face deleted successfully'})
    
    return jsonify({'error': 'Failed to delete face'}), 400

@app.route('/api/faces/gallery', methods=['GET'])
def get_face_gallery():
    """Get all registered face images"""
    gallery = []
    try:
        # Get all jpg files from saved_faces directory
        face_files = [f for f in os.listdir(detector.saved_faces_dir) 
                     if f.endswith('.jpg') and not f.startswith('.')]
        
        # Group files by name (before the timestamp)
        face_dict = {}
        for file in face_files:
            name = file.split('_')[0]  # Get name part before underscore
            if name not in face_dict:
                face_dict[name] = []
            face_dict[name].append(file)
        
        # For each person, get their most recent photo
        for name, files in face_dict.items():
            if files:
                # Sort by timestamp (newest first) and get the first one
                latest_file = sorted(files, reverse=True)[0]
                image_path = os.path.join(detector.saved_faces_dir, latest_file)
                
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        gallery.append({
                            'name': name,
                            'image': img_data
                        })
                except Exception as e:
                    print(f"Error reading image {image_path}: {str(e)}")
                    continue
                    
        return jsonify({'gallery': gallery})
    except Exception as e:
        print(f"Error getting face gallery: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/faces/<int:face_id>/image', methods=['GET'])
def get_face_image(face_id):
    """Get the image for a specific face"""
    try:
        if face_id >= len(detector.known_names):
            return jsonify({'error': 'Face not found'}), 404
        
        name = detector.known_names[face_id]
        # Get all jpg files for this name from saved_faces directory
        face_files = [f for f in os.listdir(detector.saved_faces_dir) 
                     if f.startswith(name) and f.endswith('.jpg')]
        
        if not face_files:
            return jsonify({'error': 'Image not found'}), 404
            
        # Get the most recent image
        latest_file = sorted(face_files, reverse=True)[0]
        image_path = os.path.join(detector.saved_faces_dir, latest_file)
        
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            return jsonify({'image': img_data})
            
    except Exception as e:
        print(f"Error getting face image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send current logs to the newly connected client
    try:
        logs = read_latest_logs()
        socketio.emit('initial_logs', {'logs': logs}, room=request.sid)
    except Exception as e:
        print(f"Error sending initial logs: {str(e)}")
    # Start camera
    handle_start_camera()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    # Don't stop the camera on disconnect to keep it running for other clients
    
@socketio.on('start_camera')
def handle_start_camera():
    global camera_thread
    print("Received start_camera request")
    if camera_thread is None or not camera_thread.is_alive():
        try:
            camera_running = True
            camera_thread = threading.Thread(target=camera_feed)
            camera_thread.daemon = True
            camera_thread.start()
            print("Camera thread started")
        except Exception as e:
            print(f"Error starting camera thread: {str(e)}")
    else:
        print("Camera thread already running")

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera_running
    print("Stopping camera feed")
    camera_running = False

# Update FaceDetector's log_detection method to emit events
def emit_log(message):
    """Emit a log message to all connected clients"""
    socketio.emit('new_log', {'message': message})

if __name__ == '__main__':
    # Create a templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Start ngrok
    start_ngrok()
    
    print("Starting Face Recognition Web App")
    print("--------------------------------")
    print("1. Local access: http://localhost:5000")
    print("2. Network access: http://<your-pi-ip>:5000")
    print("3. Ngrok access: Check ngrok dashboard or terminal output")
    print("--------------------------------")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    finally:
        cleanup()