#!/bin/bash

# Create a tunnel from Mac to Pi to door controller
# Local port 8082 -> Pi -> Door Controller port 80
ssh -N -R 8082:192.168.68.210:80 -p 2222 cloudwalkoffice@100.94.146.43

# The -N flag means "don't execute remote commands"
# The -R flag means "reverse tunnel"

# Keep the tunnel running in the background
# If the connection drops, the script will automatically reconnect 