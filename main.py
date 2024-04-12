import cv2
import os
import numpy as np
from datetime import datetime
from deepface import DeepFace
import tkinter as tk
from tkinter import simpledialog, messagebox

# Create a folder to save captured face images
captured_faces_folder = "captured_faces"
if not os.path.exists(captured_faces_folder):
    os.makedirs(captured_faces_folder)

# Load the image of the person
person_image = cv2.imread("chamod.jpg")

# Extract the person name from the image file name
person_name = os.path.splitext(os.path.basename("chamod.jpg"))[0]

# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces and draw rectangles around them
def detect_faces_and_draw_rectangles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image, faces

# Create a VideoCapture object to capture video from the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load features for the person image
person_features = DeepFace.represent(person_image, enforce_detection=False)
person_features_vector = person_features[0]['embedding']

# Initialize list to store face features
captured_faces_features = []

while True:
    # Initialize variables for capturing 50 face images
    capture_count = 0
    confirmation_done = False  # Variable to track if confirmation has been asked for
    name_entered = False  # Variable to track if name has been entered

    # Initialize list to store face features
    captured_faces_features = []

    while capture_count < 50:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect faces in the webcam stream
        frame_with_rectangles, faces = detect_faces_and_draw_rectangles(frame)

        # Check if any faces are detected in the webcam stream
        if len(faces) > 0:
            # Extract the region of interest (ROI) corresponding to the detected face
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]

            # Use face recognition model to extract features from the face ROI
            webcam_features = DeepFace.represent(face_roi, enforce_detection=False)
            webcam_features_vector = webcam_features[0]['embedding']

            # Compute the dot product of the feature vectors
            similarity = np.dot(webcam_features_vector, person_features_vector)

            # Set a similarity threshold
            threshold = 0.6

            # If similarity is below the threshold, and confirmation not done, prompt for confirmation
            if similarity < threshold and not confirmation_done:
                confirmation = messagebox.askyesno("Confirmation", "Do you want to capture and save your face?")
                if confirmation:
                    confirmation_done = True  # Set confirmation flag to True after asking once

            # If confirmation is done, and face is unknown, prompt for name and proceed with capturing and saving
            if confirmation_done and similarity < threshold:
                if not name_entered:
                    # Prompt user to enter name using GUI input dialog
                    root = tk.Tk()
                    root.withdraw()
                    user_name = simpledialog.askstring("Input", "Enter your name:")

                    # Create subfolder with user's name
                    subfolder_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + user_name
                    subfolder_path = os.path.join(captured_faces_folder, subfolder_name)
                    os.makedirs(subfolder_path)

                    name_entered = True  # Set name entered flag to True after asking once

                # Save the captured face image with a unique filename inside the user's folder
                image_filename = os.path.join(subfolder_path, f"face_{capture_count}.jpg")
                cv2.imwrite(image_filename, face_roi)
                capture_count += 1

                print(f"Captured face image {capture_count}")

                # Append face features to the list
                captured_faces_features.append(webcam_features_vector)

        # Display the live stream frame with detected faces
        cv2.imshow("Live Stream with Face Capture", frame_with_rectangles)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Convert list of face features to numpy array and save to file
    if len(captured_faces_features) > 0:
        captured_faces_features = np.array(captured_faces_features)
        np.save(os.path.join(subfolder_path, f"{user_name}_faces_features.npy"), captured_faces_features)
        print("Captured face images features saved to 'captured_faces_features.npy' in the subfolder.")

    # Prompt user to continue or exit
    continue_capture = messagebox.askyesno("Continue", "Do you want to continue capturing faces?")
    if not continue_capture:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
