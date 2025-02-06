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
scp -P $PI_PORT setup_network.sh $PI_USER@$PI_HOST:$PI_PATH/
scp -P $PI_PORT monitor_network.sh $PI_USER@$PI_HOST:$PI_PATH/
scp -P $PI_PORT test_door.py $PI_USER@$PI_HOST:$PI_PATH/
scp -P $PI_PORT door_proxy.py $PI_USER@$PI_HOST:$PI_PATH/
scp -P $PI_PORT -r templates $PI_USER@$PI_HOST:$PI_PATH/

# Make scripts executable and install dependencies
ssh -p $PI_PORT $PI_USER@$PI_HOST "
    chmod +x $PI_PATH/*.sh && \
    pip3 install flask-socketio eventlet python-engineio==3.14.2 python-socketio==4.6.1
"

# Update systemd service to use our new startup script
ssh -p $PI_PORT $PI_USER@$PI_HOST "
    sudo bash -c 'cat > /etc/systemd/system/face-recognition.service << EOL
[Unit]
Description=Face Recognition System
After=network.target

[Service]
Type=simple
User=$PI_USER
WorkingDirectory=$PI_PATH
Environment=PATH=$PI_PATH/face_detection_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$PI_PATH/start.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOL'

    sudo systemctl daemon-reload
    sudo systemctl enable face-recognition
    sudo systemctl restart face-recognition
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