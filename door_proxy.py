from flask import Flask, request
import requests
from requests.auth import HTTPDigestAuth
import logging
from urllib3.exceptions import InsecureRequestWarning
import urllib3
from werkzeug.serving import WSGIRequestHandler
import subprocess
import time

# Suppress only the single warning from urllib3 needed.
urllib3.disable_warnings(InsecureRequestWarning)

# Configure the WSGIRequestHandler to ignore client disconnects
class CustomRequestHandler(WSGIRequestHandler):
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionError, OSError) as e:
            # Ignore client disconnection errors
            pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Door controller settings
DOOR_CONTROLLER_IP = "127.0.0.1"  # Changed to localhost
DOOR_CONTROLLER_PORT = "8082"      # New port for tunneling
DOOR_CONTROLLER_URL = f"http://{DOOR_CONTROLLER_IP}:{DOOR_CONTROLLER_PORT}/cgi-bin/accessControl.cgi"
DOOR_USERNAME = "admin"
DOOR_PASSWORD = "9cby@GmP"

# Add cooldown tracking
last_door_action = 0
COOLDOWN_PERIOD = 4  # seconds between door actions

def check_network():
    """Check network connectivity to door controller"""
    try:
        # Try ping
        ping = subprocess.run(['ping', '-c', '1', '-W', '2', DOOR_CONTROLLER_IP], 
                            capture_output=True, text=True)
        app.logger.info(f"Ping result:\n{ping.stdout}\n{ping.stderr}")
        
        # Get network interfaces
        ifconfig = subprocess.run(['ifconfig'], capture_output=True, text=True)
        app.logger.info(f"Network interfaces:\n{ifconfig.stdout}")
        
        # Get route
        route = subprocess.run(['ip', 'route', 'get', DOOR_CONTROLLER_IP], 
                             capture_output=True, text=True)
        app.logger.info(f"Route to door controller:\n{route.stdout}")
        
    except Exception as e:
        app.logger.error(f"Error checking network: {str(e)}")

@app.route('/door')
def proxy_door():
    global last_door_action
    current_time = time.time()
    
    # Check if we're still in cooldown period
    if current_time - last_door_action < COOLDOWN_PERIOD:
        app.logger.info("Request ignored - within cooldown period")
        return "Cooldown period active", 429  # 429 = Too Many Requests
    
    action = request.args.get('action')
    channel = request.args.get('channel', '1')
    
    app.logger.info(f"Received request: action={action}, channel={channel}")
    
    try:
        app.logger.debug(f"Attempting to connect to door controller at {DOOR_CONTROLLER_URL}")
        check_network()  # Check network before attempting connection
        
        # Create a session for better connection handling
        session = requests.Session()
        session.verify = False  # Disable SSL verification
        
        # Add routing for cross-network access if needed
        session.trust_env = False  # Ignore proxy settings
        
        response = session.get(
            DOOR_CONTROLLER_URL,
            params={'action': action, 'channel': channel},
            auth=HTTPDigestAuth(DOOR_USERNAME, DOOR_PASSWORD),
            timeout=10
        )
        
        app.logger.info(f"Door controller response: {response.status_code} - {response.text}")
        
        # Update last action time only on successful request
        last_door_action = current_time
        return response.text, response.status_code
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error to door controller: {str(e)}"
        app.logger.error(error_msg)
        check_network()  # Check network after connection error
        return error_msg, 500
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout connecting to door controller: {str(e)}"
        app.logger.error(error_msg)
        check_network()  # Check network after timeout
        return error_msg, 500
    except Exception as e:
        error_msg = f"Error proxying request: {str(e)}"
        app.logger.error(error_msg)
        return error_msg, 500

@app.route('/test')
def test():
    app.logger.info("Test endpoint called")
    check_network()  # Check network on test endpoint
    return "Door proxy is running!", 200

@app.route('/')
def home():
    return """
    <h1>Door Proxy Server</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>/test - Test if server is running</li>
        <li>/door?action=openDoor&channel=1 - Open door</li>
        <li>/door?action=closeDoor&channel=1 - Close door</li>
    </ul>
    """

if __name__ == '__main__':
    print("Starting door proxy server on port 8081...")
    print(f"Door controller URL: {DOOR_CONTROLLER_URL}")
    print("Test the server with: curl http://localhost:8081/test")
    
    # Initial network check
    check_network()
    
    # Run without debug mode and use the custom request handler
    app.run(host='0.0.0.0', 
            port=8081, 
            debug=False,
            request_handler=CustomRequestHandler)
