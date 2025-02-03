#!/bin/bash

# Pi connection details
PI_USER="cloudwalkoffice"
PI_HOST="100.94.146.43"
PI_PORT="2222"
PI_PATH="~/face_detection"

# Transfer files
echo "Transferring files to Raspberry Pi..."
scp -P $PI_PORT face_detector.py $PI_USER@$PI_HOST:$PI_PATH/
scp -P $PI_PORT web_app.py $PI_USER@$PI_HOST:$PI_PATH/
scp -P $PI_PORT start.sh $PI_USER@$PI_HOST:$PI_PATH/
scp -P $PI_PORT -r templates $PI_USER@$PI_HOST:$PI_PATH/

echo "Setting up Face Recognition System on Raspberry Pi..."
echo "=================================================="

# SSH into Pi and run setup commands
ssh -p $PI_PORT $PI_USER@$PI_HOST "
    echo 'Updating system packages...' && \
    sudo apt update && \
    sudo apt upgrade -y && \
    
    echo 'Installing system dependencies...' && \
    sudo apt install -y \
        python3-full \
        python3-pip \
        python3-venv \
        libcamera-tools \
        cmake \
        build-essential \
        libopencv-dev \
        python3-opencv \
        git && \
    
    echo 'Setting up application directory...' && \
    mkdir -p $PI_PATH && \
    
    echo 'Setting up Python virtual environment...' && \
    cd $PI_PATH && \
    python3 -m venv face_detection_env && \
    source face_detection_env/bin/activate && \
    
    echo 'Installing Python packages...' && \
    pip install --upgrade pip && \
    pip install \
        face_recognition \
        opencv-python \
        flask \
        flask-socketio \
        pillow \
        numpy \
        dlib && \
    
    echo 'Creating application directories...' && \
    mkdir -p $PI_PATH/saved_faces && \
    mkdir -p $PI_PATH/detection_logs && \
    mkdir -p $PI_PATH/templates && \
    mkdir -p $PI_PATH/new_registrations && \
    
    echo 'Making scripts executable...' && \
    chmod +x $PI_PATH/start.sh && \
    
    echo 'Creating systemd service...' && \
    sudo bash -c 'cat > /etc/systemd/system/face-recognition.service << EOL
[Unit]
Description=Face Recognition System
After=network.target

[Service]
Type=simple
User=$PI_USER
WorkingDirectory=$PI_PATH
Environment=PATH=$PI_PATH/face_detection_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$PI_PATH/face_detection_env/bin/python web_app.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOL' && \
    
    echo 'Enabling and starting service...' && \
    sudo systemctl daemon-reload && \
    sudo systemctl enable face-recognition && \
    sudo systemctl start face-recognition && \
    
    echo 'Checking service status...' && \
    sudo systemctl status face-recognition
"

echo "=================================================="
echo "Deployment completed!"
echo ""
echo "The application should now be running on your Raspberry Pi"
echo "To check status: ssh into the Pi and run 'sudo systemctl status face-recognition'"
echo "To view logs: ssh into the Pi and run 'sudo journalctl -u face-recognition -f'"
echo ""
echo "The application will be available at:"
echo "- Local access: http://localhost:5000"
echo "- Network access: http://<your-pi-ip>:5000"
echo "==================================================" 