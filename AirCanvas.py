# AirCanvas.py
# Author: Amir Ghavam
# Description: A simple air drawing application using OpenCV and MediaPipe.

# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set desired frame width and height
FRAME_WIDTH = 1280
Frame_HEIGHT = 720

# Set width and height of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Frame_HEIGHT)

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
canvas = np.zeros((Frame_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


# Define color buttons and eraser button
buttons_colors_eraser = {
    "Blue":    {"rect": (20, 40, 120, 80),   "color": (255, 0, 0)},
    "Red":     {"rect": (20, 95, 120, 135),  "color": (0, 0, 255)},
    "Green":   {"rect": (20, 150, 120, 190), "color": (0, 255, 0)},
    "White":   {"rect": (20, 205, 120, 245), "color": (255, 255, 255)},
    "Yellow":  {"rect": (20, 260, 120, 300), "color": (0, 255, 255)},
    "Pink":    {"rect": (20, 315, 120, 355), "color": (255, 0, 255)},
    "Eraser":  {"rect": (20, 405, 120, 445), "color": (0, 0, 0)}
}

current_color = (255, 0, 0)  # set Green default
brush_size = 8 # set 8 as default brush size


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
            cv2.circle(frame, (x, y), 10, current_color, -1)

        # Initialize previous point on first detection
        if prev_x is None:
            prev_x, prev_y = x, y


        # Check if fingertip touches any button
        for name, btn in buttons_colors_eraser.items():
            x1, y1, x2, y2 = btn["rect"]

            if x1-1 < x < x2+1 and y1-1 < y < y2+1:
                # Change current color or eraser
                current_color = btn["color"]

        # Draw on canvas
        # Only draw if fingertip is outside the UI area (+ some margin)
        if x > 130:    
            cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, brush_size)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None


    else:
        # If no hand detected, reset previous point
        prev_x, prev_y = None, None

    # Define text properties
    TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX  
    FONT_SCALE = 0.75  
    TEXT_THICKNESS = 1  


    # Draw UI buttons
    for name, btn in buttons_colors_eraser.items():
        x1, y1, x2, y2 = btn["rect"]
        
        # The rect coordinates for the border should be slightly larger
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 2)
        
        # Draw the main button rectangle (filled)
        cv2.rectangle(frame, (x1, y1), (x2, y2), btn["color"], -1)
        
        # Determine text color based on button color
        if name.lower() == "white" or name.lower() == "yellow":
            text_color = (0, 0, 0)  # Black text for light colors
        else:
            text_color = (255, 255, 255)  # White text for dark colors
            
        # Add text label to button
        cv2.putText(frame, name.upper(), (x1 + 5, y1 + 30),
                    TEXT_FONT, FONT_SCALE,
                    text_color, TEXT_THICKNESS, cv2.LINE_AA) # Use cv2.LINE_AA for anti-aliasing the text!

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
