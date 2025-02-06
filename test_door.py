from face_detector import FaceDetector
import subprocess
import socket

def check_network():
    try:
        print("\nNetwork Diagnostics:")
        # Get Pi's IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        pi_ip = s.getsockname()[0]
        s.close()
        print(f"Pi's IP address: {pi_ip}")
        
        # Print all network interfaces
        print("\nNetwork Interfaces:")
        ifconfig = subprocess.run(['ifconfig'], capture_output=True, text=True)
        print(ifconfig.stdout)
        
        door_ip = "192.168.68.210"
        print(f"\nTesting connection to door controller ({door_ip}):")
        
        # Print route to door controller
        print("\nRoute to door controller:")
        route = subprocess.run(['ip', 'route', 'get', door_ip], capture_output=True, text=True)
        print(route.stdout)
        
        # Try ping with verbose output
        print("\nPing test:")
        ping_result = subprocess.run(['ping', '-v', '-c', '1', '-W', '2', door_ip], 
                                   capture_output=True, text=True)
        print(ping_result.stdout)
        print(ping_result.stderr)
        
        # Try curl with more verbose output
        print("\nTesting HTTP connection:")
        curl_result = subprocess.run([
            'curl', '-v', '--digest', 
            '-u', 'admin:9cby@GmP',
            '-m', '5',  # 5 second timeout
            '--interface', 'tailscale0',  # Try using Tailscale interface
            f'http://{door_ip}/cgi-bin/accessControl.cgi?action=openDoor&channel=1'
        ], capture_output=True, text=True)
        print("Curl output:")
        print(curl_result.stdout)
        print("Curl error output:")
        print(curl_result.stderr)
        
        # Print all routes
        print("\nAll routes:")
        routes = subprocess.run(['ip', 'route', 'show'], capture_output=True, text=True)
        print(routes.stdout)
        
    except Exception as e:
        print(f"Error during network diagnostics: {str(e)}")

def test_door():
    print("Running network diagnostics first...")
    check_network()
    
    print("\nTesting door controller...")
    detector = FaceDetector()
    
    print("\nTesting door open...")
    if detector.open_door():
        print("Door opened successfully")
        print("Waiting 5 seconds...")
        import time
        time.sleep(5)
        print("Testing door close...")
        if detector.close_door():
            print("Door closed successfully")
        else:
            print("Failed to close door")
    else:
        print("Failed to open door")

if __name__ == "__main__":
    test_door() 