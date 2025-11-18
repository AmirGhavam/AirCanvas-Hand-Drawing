# AirCanvas.py
# Author: Amir Ghavam
# Description: A simple air drawing application using OpenCV and MediaPipe.

# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from itertools import chain


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set desired frame width and height
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Set width and height of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

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
canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

# Define color buttons and eraser button
buttons_colors_eraser = {
    "Blue":    {"rect": (20, 40, 130, 80),   "color": (255, 0, 0),   "hover": False, "display_color": (255, 0, 0), "display_rect": (20, 40, 130, 80),"clicked": False},
    "Red":     {"rect": (20, 95, 130, 135),  "color": (0, 0, 255),   "hover": False, "display_color": (0, 0, 255), "display_rect": (20, 95, 130, 135),"clicked": False},
    "Green":   {"rect": (20, 150, 130, 190), "color": (0, 255, 0),   "hover": False, "display_color": (0, 255, 0), "display_rect": (20, 150, 130, 190),"clicked": False},
    "White":   {"rect": (20, 205, 130, 245), "color": (255, 255, 255),"hover": False, "display_color": (255, 255, 255),"display_rect": (20, 205, 130, 245),"clicked": False},
    "Yellow":  {"rect": (20, 260, 130, 300), "color": (0, 255, 255), "hover": False, "display_color": (0, 255, 255), "display_rect": (20, 260, 130, 300),"clicked": False},
    "Pink":    {"rect": (20, 315, 130, 355), "color": (255, 0, 255), "hover": False, "display_color": (255, 0, 255), "display_rect": (20, 315, 130, 355),"clicked": False},
    "Eraser":  {"rect": (20, 370, 130, 410), "color": (0, 0, 0),     "hover": False, "display_color": (0, 0, 0),     "display_rect": (20, 370, 130, 410),"clicked": False}
}

# # Define action buttons
buttons_actions = {
    "Clear":      {"rect": (FRAME_WIDTH - 130, 40,  FRAME_WIDTH - 20,  80),  "color": (51, 0, 51),  "hover": False, "display_color": (51, 0, 51),  "display_rect": (FRAME_WIDTH - 130, 40,  FRAME_WIDTH - 20,  80),"clicked": False},
    "Save":       {"rect": (FRAME_WIDTH - 130, 95,  FRAME_WIDTH - 20, 135),  "color": (76, 0, 102),  "hover": False, "display_color": (76, 0, 102),  "display_rect": (FRAME_WIDTH - 130, 95,  FRAME_WIDTH - 20, 135),"clicked": False},
    "Screenshot": {"rect": (FRAME_WIDTH - 130, 150, FRAME_WIDTH - 20, 190),  "color": (102, 0, 153),  "hover": False, "display_color": (102, 0, 153),  "display_rect": (FRAME_WIDTH - 130, 150, FRAME_WIDTH - 20, 190),"clicked": False},
    "ToggleCam":  {"rect": (FRAME_WIDTH - 130, 205, FRAME_WIDTH - 20, 245),  "color": (127, 0, 204), "hover": False, "display_color": (127, 0, 204), "display_rect": (FRAME_WIDTH - 130, 205, FRAME_WIDTH - 20, 245),"clicked": False},
    "Exit":       {"rect": (FRAME_WIDTH - 130, 260, FRAME_WIDTH - 20, 300),  "color": (153, 0, 255),   "hover": False, "display_color": (153, 0, 255),   "display_rect": (FRAME_WIDTH - 130, 260, FRAME_WIDTH - 20, 300),"clicked": False}
}

# Set default drawing color and brush size
current_color = (255, 0, 0)  
brush_size = 8 

# Camera toggle state
camera_enabled = True

# Main loop
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

    # Prepare display frame based on camera toggle
    if camera_enabled:
        display_frame = frame.copy()
    else:
        display_frame = np.zeros_like(frame)


    # Process frame for hand detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Loop through detected hands (we're using max 1 hand)
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame 
            mp_drawing.draw_landmarks(
                display_frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # GET INDEX FINGER TIP 
            index_tip = hand_landmarks.landmark[8]

            # Convert from normalized → pixel coordinates 
            x = int(index_tip.x * FRAME_WIDTH) 
            y = int(index_tip.y * FRAME_HEIGHT)

            # Draw a circle on the index tip
            cv2.circle(display_frame, (x, y), 10, current_color, -1)

        # Initialize previous point on first detection
        if prev_x is None:
            prev_x, prev_y = x, y


        # Check hover state for all buttons and update button appearance
        for name, btn in chain(buttons_colors_eraser.items(), buttons_actions.items()):
            x1, y1, x2, y2 = btn["rect"]
            inside = (x1-1 < x < x2+1 and y1-1 < y < y2+1)

            if inside and not btn["hover"]:
                # ENTER event → darken + pop-out
                btn["hover"] = True

                # darken color
                c = btn["color"]
                btn["display_color"] = (int(c[0] * 0.6), int(c[1] * 0.6), int(c[2] * 0.6))

                # expand rect (pop-out effect)
                btn["display_rect"] = (x1 - 2, y1 - 2, x2 + 2, y2 + 2)

            elif not inside and btn["hover"]:
                # EXIT event → restore
                btn["hover"] = False
                btn["clicked"] = False
                btn["display_color"] = btn["color"]
                btn["display_rect"] = btn["rect"]


        # Handle color and eraser buttons
        for name, btn in buttons_colors_eraser.items():
            x1, y1, x2, y2 = btn["rect"]
            inside = (x1-1 < x < x2+1 and y1-1 < y < y2+1)

            # If fingertip is inside AND hover is True AND we haven't fired yet -> fire once
            if inside and btn["hover"] and not btn.get("clicked", False):
                btn["clicked"] = True  # prevent repeated firing while finger stays inside
                # Change current color or eraser
                current_color = btn["color"]

        for name, btn in buttons_actions.items():
            x1, y1, x2, y2 = btn["rect"]
            inside = (x1-1 < x < x2+1 and y1-1 < y < y2+1)

            # If fingertip is inside AND hover is True AND we haven't fired yet -> fire once
            if inside and btn["hover"] and not btn.get("clicked", False):
                btn["clicked"] = True  # prevent repeated firing while finger stays inside

                if name == "Clear":
                    canvas[:] = 0
                    print("Canvas cleared")

                elif name == "Save":
                    cv2.imwrite("drawing.png", canvas)
                    print("Saved drawing.png")

                elif name == "Screenshot":
                    # make sure 'combined' exists (blend done later), or build it here:
                    combined_img = cv2.addWeighted(display_frame, 0.5, canvas, 1, 0) if camera_enabled else canvas.copy()
                    cv2.imwrite("screenshot.png", combined_img)
                    print("Saved screenshot.png")

                elif name == "ToggleCam":
                    camera_enabled = not camera_enabled
                    print("Camera enabled:", camera_enabled)

                elif name == "Exit":
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        # Draw on canvas
        if x > 140 and x < FRAME_WIDTH - 140:    
            cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, brush_size)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None

    else:
        # If no hand detected, reset previous point
        prev_x, prev_y = None, None

    # Draw buttons
    for name, btn in chain(buttons_colors_eraser.items(), buttons_actions.items()):
        x1, y1, x2, y2 = btn["display_rect"]
        color = btn["display_color"]

        # Determine text and border color for contrast
        if sum(color) > 400:
            color_text_border = (0, 0, 0)
        else:
            color_text_border = (255, 255, 255)

        # Border
        cv2.rectangle(display_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color_text_border, 2)

        # Button fill
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, -1)

        # Draw text
        cv2.putText(display_frame, name.upper(), (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color_text_border, 1, cv2.LINE_AA)

    # Combine canvas and video
    combined = cv2.addWeighted(display_frame, 0.5, canvas, 1, 0)

    # Show the combined image
    cv2.imshow("Drawing Canvas", combined)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
