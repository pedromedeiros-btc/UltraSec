# CWUltrasec

A Raspberry Pi-based real-time face recognition system with adaptive learning capabilities.

## Features

* 🎭 Real-time face detection & recognition
* 📸 Automatic face enrollment via center-targeting system
* 🧠 Adaptive learning with face encoding storage
* 📊 Live confidence metrics in debug interface
* 🔍 Optimized for Raspberry Pi 4 performance
* 📁 Automatic data persistence (images + encodings)

## Hardware Requirements

* Raspberry Pi 4 (8GB recommended)
* Raspberry Pi Camera Module (or USB webcam)
* 5V 3A Power Supply
* MicroSD Card (32GB+ recommended)
* Active cooling solution (recommended)

## Software Dependencies

* Python 3.7+
* OpenCV (opencv-python)
* face_recognition library
* NumPy
* Pickle (built-in)
* datetime (built-in)

## Installation

1. Clone Repository:
```bash
git clone https://github.com/pedromedeiros-btc/UltraSec
cd UltraSec
```

2. Install Dependencies:
```bash
pip install -r requirements.txt
```

3. For Raspberry Pi Camera setup:
```bash
sudo apt-get install python3-picamera python3-picamera[array]
sudo raspi-config  # Navigate to: Interfacing Options > Camera > Enable Usage
```

## Usage

Run the main application:
```bash
python face_detector.py
```

### Interface Controls
* S: Save face in detection square
* Q: Quit application
* Mouse: Adjust window positions

### Visual Guide
* 🟩 Green Square: Face enrollment area
* 🔴 Red Border: Face ready for enrollment
* 🔵 Blue Border: Recognized face
* 📊 Debug Window: Shows recognition confidence metrics

## Configuration

### Recognition sensitivity (lower = stricter)
```python
RECOGNITION_THRESHOLD = 0.6
```

### Detection square parameters (percentage of screen)
```python
SQUARE_SIZE_RATIO = 0.5  # 50% of screen size
```

### Storage locations
```python
SAVED_FACES_DIR = "saved_faces"
ENCODINGS_FILE = "known_faces.pkl"
```

## Ethical Considerations

**Important Notice:** This system should be used in compliance with all local privacy laws and regulations. We recommend:
* Clear signage when surveillance is active
* Secure storage of facial data
* Regular data audits
* Explicit consent for enrolled faces
* Performance optimization for accuracy

## License

[Add your license information here]
