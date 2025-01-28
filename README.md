UltraSec
A Raspberry Pi-based real-time face recognition system with adaptive learning capabilities
 (Replace with actual screenshot/video)
Features
:performing_arts: Real-time face detection & recognition
:camera_with_flash: Automatic face enrollment via center-targeting system
:brain: Adaptive learning with face encoding storage
:bar_chart: Live confidence metrics in debug interface
:mag: Optimized for Raspberry Pi 4 performance
:file_folder: Automatic data persistence (images + encodings)
Hardware Requirements
Raspberry Pi 4 (8GB recommended)
Raspberry Pi Camera Module (or USB webcam)
5V 3A Power Supply
MicroSD Card (32GB+ recommended)
Active cooling solution (recommended)
Software Dependencies
Python 3.7+
OpenCV (opencv-python)
face_recognition library
NumPy
Pickle (built-in)
datetime (built-in)
Installation
Clone Repository
bash
Copy
git clone https://github.com/yourusername/pisurveillance-ai.git
cd pisurveillance-ai
Install Dependencies
bash
Copy
pip install -r requirements.txt
For Raspberry Pi Camera setup:
bash
Copy
sudo apt-get install python3-picamera python3-picamera[array]
Enable Camera Interface
bash
Copy
sudo raspi-config
Navigate to: Interfacing Options > Camera > Enable
Usage
bash
Copy
python surveillance_main.py
Interface Controls:
S: Save face in detection square
Q: Quit application
Mouse: Adjust window positions
Visual Guide:
:large_green_square: Green Square: Face enrollment area
:red_circle: Red Border: Face ready for enrollment
:large_blue_circle: Blue Border: Recognized face
:bar_chart: Debug Window: Shows recognition confidence metrics
Configuration
Modify these values in surveillance_main.py for customization:
python
Copy
# Recognition sensitivity (lower = stricter)
RECOGNITION_THRESHOLD = 0.6

# Detection square parameters (percentage of screen)
SQUARE_SIZE_RATIO = 0.5  # 50% of screen size

# Storage locations
SAVED_FACES_DIR = "saved_faces"
ENCODINGS_FILE = "known_faces.pkl"
Ethical Considerations
:warning: Important Notice:
This system should be used in compliance with all local privacy laws and regulations. We recommend:
Clear signage when surveillance is active
Secure storage of facial data
Regular data audits
Explicit consent for enrolled faces
Performance Tips
Use dedicated Pi Camera module over USB webcam
Close unnecessary background processes
Maintain operating temperature below 60°C
Reduce frame size in code for higher FPS:
python
Copy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
Troubleshooting
Camera Not Detected:
bash
Copy
sudo modprobe bcm2835-v4l2
Low Recognition Accuracy:
Ensure front-facing, well-lit face enrollment
Multiple enrollment angles recommended
Clean camera lens
High CPU Usage:
Reduce processing frame rate by increasing:
python
Copy
process_this_frame = not process_this_frame  # Change to skip more frames
License
This project is released under the MIT License.
Important Note: While open-source, commercial use requires additional permissions. Contact author for enterprise licensing.
Future Roadmap
Multi-face simultaneous enrollment
Network streaming capabilities
Temperature monitoring integration
Mask detection module
EdgeTPU acceleration support
Disclaimer: This project is intended for educational and authorized security purposes only. Developers are not responsible for misuse or unauthorized surveillance activities.
