import cv2
import numpy as np
import os

def detect_eyes(image_path="image_two.png", use_webcam=False):
    """
    Detect eyes in an image or webcam feed using Haar Cascade Classifiers.
    
    Args:
        image_path (str): Path to the input image
        use_webcam (bool): Whether to use webcam input
    """
    # Create output directory if it doesn't exist
    output_dir = "../output/eye_detection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pre-trained classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if use_webcam:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            processed_frame = process_frame(frame, face_cascade, eye_cascade)
            cv2.imwrite(os.path.join(output_dir, "eye_detection_result.jpg"), processed_frame)
        cap.release()
    else:
        if image_path is None or not os.path.exists(image_path):
            raise ValueError(f"Please provide a valid image path. Could not find {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        processed_image = process_frame(image, face_cascade, eye_cascade)
        
        # Save the processed image
        cv2.imwrite(os.path.join(output_dir, "eye_detection_result.jpg"), processed_image)

def process_frame(frame, face_cascade, eye_cascade):
    """
    Process a single frame to detect faces and eyes.
    
    Args:
        frame: Input frame/image
        face_cascade: Pre-trained face cascade classifier
        eye_cascade: Pre-trained eye cascade classifier
    
    Returns:
        Processed frame with detected eyes marked
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    # For each face, detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            center = (x + ex + ew//2, y + ey + eh//2)
            radius = min(ew, eh) // 2
            cv2.circle(frame, center, radius, (0, 0, 255), 2)
    
    return frame

if __name__ == "__main__":
    # Example usage with default image
    detect_eyes() 