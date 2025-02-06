from face_detector import FaceDetector
import cv2
import time

def test_face_recognition_door():
    detector = FaceDetector()
    
    print("Starting face recognition test...")
    print("Press 'q' to quit")
    
    while True:
        success, frame = detector.get_frame_with_detections()
        if success:
            # Display the frame
            cv2.imshow('Face Recognition Test', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to get frame")
            break
        
        time.sleep(0.1)  # Small delay to prevent CPU overuse
    
    cv2.destroyAllWindows()
    detector.cleanup()

if __name__ == "__main__":
    test_face_recognition_door() 