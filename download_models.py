import os
import bz2
import requests

def download_and_extract_model(url, filename):
    print(f"Downloading {filename}...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Download the file
    response = requests.get(url, allow_redirects=True)
    compressed_path = f"models/{filename}.bz2"
    
    # Save the compressed file
    with open(compressed_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the file
    print(f"Extracting {filename}...")
    with bz2.BZ2File(compressed_path) as fr, open(f"models/{filename}", 'wb') as fw:
        fw.write(fr.read())
    
    # Remove the compressed file
    os.remove(compressed_path)
    print(f"Successfully downloaded and extracted {filename}")

# URLs for the model files
urls = {
    "shape_predictor_68_face_landmarks.dat": 
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat":
        "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
}

for filename, url in urls.items():
    if not os.path.exists(f"models/{filename}"):
        download_and_extract_model(url, filename)
    else:
        print(f"{filename} already exists")