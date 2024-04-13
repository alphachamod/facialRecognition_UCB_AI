import cv2
import os
import numpy as np
from datetime import datetime
from deepface import DeepFace
import tkinter as tk
from tkinter import messagebox, simpledialog

# Create a folder to save captured face images
captured_faces_folder = "captured_faces"
if not os.path.exists(captured_faces_folder):
    os.makedirs(captured_faces_folder)

# Load the saved numpy files containing the facial features of each person
saved_faces_features = {}
for folder_name in os.listdir(captured_faces_folder):
    folder_path = os.path.join(captured_faces_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith("_faces_features.npy"):
                name = file_name.split("_")[0]
                features = np.load(os.path.join(folder_path, file_name))
                saved_faces_features[name] = features

# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces and draw rectangles around them
def detect_faces_and_draw_rectangles(image, names):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the detected face
        face_roi = image[y:y+h, x:x+w]

        # Use face recognition model to extract features from the face ROI
        webcam_features = DeepFace.represent(face_roi, enforce_detection=False)
        webcam_features_vector = webcam_features[0]['embedding']

        # Initialize variables for matching
        max_similarity = 0
        matched_name = "Unknown"

        # Compare the features of the detected face with the features of each person in the saved numpy files
        for name, saved_features in saved_faces_features.items():
            for saved_feature in saved_features:
                similarity = np.dot(webcam_features_vector, saved_feature)
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_name = name

        # Draw rectangle around the face and display the name
        threshold = 0.6
        if max_similarity > threshold:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{matched_name} ({max_similarity:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Prompt a confirmation window if the face is unknown
            confirmation = messagebox.askyesno("Confirmation", "Unknown face detected. Add as a new person?")
            if confirmation:
                # Prompt for name input
                user_name = simpledialog.askstring("Name", "Please enter your name:")
                if user_name:
                    # Create a subfolder inside the "captured_faces" folder
                    subfolder_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + user_name
                    subfolder_path = os.path.join(captured_faces_folder, subfolder_name)
                    os.makedirs(subfolder_path)

                    # Save the captured face images with a unique filename
                    capture_count = 0
                    while capture_count < 100:
                        cv2.imwrite(os.path.join(subfolder_path, f"face_{capture_count}.jpg"), face_roi)
                        capture_count += 1

                    # Save the features of captured face images to a numpy file
                    captured_faces_features = []
                    for filename in os.listdir(subfolder_path):
                        if filename.endswith(".jpg"):
                            image_path = os.path.join(subfolder_path, filename)
                            face_image = cv2.imread(image_path)
                            face_features = DeepFace.represent(face_image, enforce_detection=False)
                            captured_faces_features.append(face_features[0]['embedding'])

                    captured_faces_features = np.array(captured_faces_features)
                    np.save(os.path.join(subfolder_path, f"{user_name}_faces_features.npy"), captured_faces_features)

                    # Display success message
                    messagebox.showinfo("Success", "Face captured and saved successfully.")

    return image

# Create a VideoCapture object to capture video from the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Detect faces in the webcam stream and draw rectangles around them
    frame_with_rectangles = detect_faces_and_draw_rectangles(frame, saved_faces_features.keys())

    # Display the live stream frame with detected faces and names
    cv2.imshow("Visitor Screening System UCB Assignment", frame_with_rectangles)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
