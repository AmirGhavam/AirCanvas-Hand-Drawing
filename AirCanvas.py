# AirCanvas.py
# Author: Amir Ghavam
# Description: A simple air drawing application using OpenCV and MediaPipe.

# Import necessary libraries
import cv2
import mediapipe as mp

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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# if not cap.isOpened():
#     print("ERROR: Could not access the camera.")
# else:
#     print("Camera is working!")

while True:
    ret, frame = cap.read()
    if not ret:
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

    # Display the frame
    cv2.imshow("Air Drawing", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()