# AirCanvas.py
# Author: Amir Ghavam
# Description: A simple air drawing application using OpenCV and MediaPipe.

# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set width and height of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # We only need to track one hand for drawing
    min_detection_confidence=0.7, # Adjusted for better detection
    min_tracking_confidence=0.5 # Adjusted for better tracking
)
mp_drawing = mp.solutions.drawing_utils

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("ERROR: Could not access the camera.")
else:
    print("Camera is working!")

# Store last index finger position
prev_x, prev_y = None, None   

# Create a black canvas same size as webcam
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret: # Check if frame is grabbed
        print("Failed to grab frame")
        break

    # Mirror the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)
    # MediaPipe needs RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    # Process frame for hand detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Loop through detected hands (we're using max 1 hand)
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame 
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # GET INDEX FINGER TIP 
            index_tip = hand_landmarks.landmark[8]

            # Convert from normalized → pixel coordinates 
            h, w, _ = frame.shape
            x = int(index_tip.x * w) 
            y = int(index_tip.y * h)

            # Draw a circle on the index tip
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

        # Initialize previous point on first detection
        if prev_x is None:
            prev_x, prev_y = x, y

        # Draw on canvas
        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 8)

        # Update previous point
        prev_x, prev_y = x, y
    else:
        # If no hand detected, reset previous point
        prev_x, prev_y = None, None

     # Combine canvas and video
    combined = cv2.addWeighted(frame, 0.5, canvas, 1, 0)

    # Show the combined image
    cv2.imshow("Drawing Canvas", combined)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
