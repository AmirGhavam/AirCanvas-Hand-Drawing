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
    "Eraser":  {"rect": (20, 520, 130, 560), "color": (0, 0, 0),     "hover": False, "display_color": (0, 0, 0),     "display_rect": (20, 520, 130, 560),"clicked": False}
}

# # Define action buttons
buttons_actions = {
    "Clear":      {"rect": (FRAME_WIDTH - 130, 340,  FRAME_WIDTH - 20,  380),  "color": (51, 0, 51),  "hover": False, "display_color": (51, 0, 51),  "display_rect": (FRAME_WIDTH - 130, 340,  FRAME_WIDTH - 20,  380),"clicked": False},
    "Save":       {"rect": (FRAME_WIDTH - 130, 495,  FRAME_WIDTH - 20, 435),  "color": (76, 0, 102),  "hover": False, "display_color": (76, 0, 102),  "display_rect": (FRAME_WIDTH - 130, 395,  FRAME_WIDTH - 20, 435),"clicked": False},
    "Screenshot": {"rect": (FRAME_WIDTH - 130, 450, FRAME_WIDTH - 20, 490),  "color": (102, 0, 153),  "hover": False, "display_color": (102, 0, 153),  "display_rect": (FRAME_WIDTH - 130, 450, FRAME_WIDTH - 20, 490),"clicked": False},
    "ToggleCam":  {"rect": (FRAME_WIDTH - 130, 505, FRAME_WIDTH - 20, 545),  "color": (127, 0, 204), "hover": False, "display_color": (127, 0, 204), "display_rect": (FRAME_WIDTH - 130, 505, FRAME_WIDTH - 20, 545),"clicked": False},
    "Exit":       {"rect": (FRAME_WIDTH - 130, 560, FRAME_WIDTH - 20, 600),  "color": (153, 0, 255),   "hover": False, "display_color": (153, 0, 255),   "display_rect": (FRAME_WIDTH - 130, 560, FRAME_WIDTH - 20, 600),"clicked": False},
    " -": {"rect": (20, 430, 65, 470), "color": (60, 60, 60), "hover": False, "display_color": (60, 60, 60), "display_rect": (20, 430, 65, 470)},
    " +": {"rect": (85, 430, 130, 470), "color": (80, 80, 80), "hover": False, "display_color": (80, 80, 80), "display_rect": (85, 430, 130, 470)},
    "Neon":      {"rect": (20, 575, 130, 615), "color": (255, 255, 255), "hover": False, "display_color": (255, 255, 255), "display_rect": (20, 575, 130, 615),"clicked": False},
    "Calligraphy": {"rect": (20, 630, 130, 670), "color": (192, 192, 192), "hover": False, "display_color": (192, 192, 192), "display_rect": (20, 630, 130, 670),"clicked": False},
    "Spray":     {"rect": (20, 685, 130, 725), "color": (128, 128, 128), "hover": False, "display_color": (128, 128, 128), "display_rect": (20, 685, 130, 725),"clicked": False},
    "Marker":    {"rect": (20, 740, 130, 780), "color": (64, 64, 64), "hover": False, "display_color": (64, 64, 64), "display_rect": (20, 740, 130, 780),"clicked": False},

}

def draw_neon_line(canvas, prev_x, prev_y, x, y, current_color, brush_size):
    r, g, b = map(int, current_color)
    
    # Create temporary overlay
    glow = np.zeros_like(canvas)
    
    sizes = [brush_size + 30, brush_size + 20, brush_size + 12, brush_size + 6, brush_size]
    colors = [
        (r*0.2, g*0.2, b*0.25),
        (r*0.4, g*0.4, b*0.5),
        (r*0.7, g*0.7, b*0.9),
        (r*0.9, g*0.9, b*1.2),
        current_color
    ]
    
    for size, color in zip(sizes, colors):
        cv2.line(glow, (prev_x, prev_y), (x, y), color, size, cv2.LINE_AA)
    
    # White hot core
    cv2.line(glow, (prev_x, prev_y), (x, y), (200, 230, 255), max(2, brush_size//3), cv2.LINE_AA)
    
    # Screen blend (true neon look)
    canvas[:] = np.maximum(canvas, glow)


def draw_caligraphy_line(canvas, prev_x, prev_y, x, y, current_color, brush_size):
    # Real calligraphy: thick when going down, thin when going sideways/up
    dx = x - prev_x
    dy = y - prev_y
    angle = np.arctan2(dy, dx)  # direction of stroke
    
    # Make it thickest when pen is moving downward (like real dip pen)
    vertical_weight = abs(np.cos(angle))  # 1.0 when vertical, 0 when horizontal
    thickness = int(brush_size * (0.3 + vertical_weight * 2.0))  # from ~0.3x to 2.3x
    thickness = max(2, min(thickness, brush_size * 4))
    
    cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness, 
            cv2.LINE_AA)  # anti-aliased = beautiful edges


def draw_spray_line(canvas, prev_x, prev_y, x, y, current_color, brush_size):
    # Proper airbrush/spray effect — dense in center, fades out
    num_dots = int(brush_size * 3)  # more dots = denser spray
    for _ in range(num_dots):
        # Gaussian distribution = natural spray
        while True:
            offset_x = np.random.randint(-brush_size*2, brush_size*2)
            offset_y = np.random.randint(-brush_size*2, brush_size*2)
            dist = (offset_x**2 + offset_y**2) ** 0.5
            if dist <= brush_size * 2:
                # Density falloff (center = full, edge = rare)
                if np.random.random() > dist / (brush_size * 2):
                    break
        
        # Slight transparency effect by varying alpha
        alpha = max(0.1, 1.0 - dist / (brush_size * 2))
        spray_color = (
            int(current_color[0] * alpha),
            int(current_color[1] * alpha),
            int(current_color[2] * alpha)
        )
        cv2.circle(canvas, (x + offset_x, y + offset_y), 
                np.random.randint(1, 4), spray_color, -1)

def draw_marker_line(canvas, prev_x, prev_y, x, y, current_color, brush_size):
    # Create temporary overlay
    overlay = canvas.astype(np.float32)
    
    # Soft glowing edge
    cv2.line(overlay, (prev_x, prev_y), (x, y), (*current_color, 70), 
            brush_size * 2, cv2.LINE_AA)
    
    # Medium layer
    cv2.line(overlay, (prev_x, prev_y), (x, y), (*current_color, 120), 
            round(brush_size * 1.3), cv2.LINE_AA)
    
    # Sharp core
    cv2.line(overlay, (prev_x, prev_y), (x, y), current_color, 
            max(2, brush_size//2), cv2.LINE_AA)
    
    # Blend with screen-like effect
    canvas[:] = np.minimum(255, overlay + canvas * 0.15)



# Set default drawing color and brush size
current_color = (255, 0, 0)  
brush_size = 8 
max_brush = 60
min_brush = 1

# Camera toggle state
camera_enabled = True

# Drawing pattern state
current_pattern = "normal"

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

                if name == " -":
                    brush_size = max(min_brush, brush_size - 1)

                elif name == " +":
                    brush_size = min(max_brush, brush_size + 1)
                
                elif name == "Neon":
                    current_pattern = "neon"

                elif name == "Calligraphy":
                    current_pattern = "calligraphy"

                elif name == "Spray":
                    current_pattern = "spray"

                elif name == "Marker":
                    current_pattern = "marker"

                elif name == "Clear":
                    canvas[:] = 0

                elif name == "Save":
                    cv2.imwrite("drawing.png", canvas)

                elif name == "Screenshot":
                    # make sure 'combined' exists (blend done later), or build it here:
                    combined_img = cv2.addWeighted(display_frame, 0.5, canvas, 1, 0) if camera_enabled else canvas.copy()
                    cv2.imwrite("screenshot.png", combined_img)

                elif name == "ToggleCam":
                    camera_enabled = not camera_enabled

                elif name == "Exit":
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()



        if 152 < x < FRAME_WIDTH - 152:

            if current_pattern == "normal":
                cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, brush_size)

            elif current_pattern == "neon":
                draw_neon_line(canvas, prev_x, prev_y, x, y, current_color, brush_size)


            elif current_pattern == "calligraphy":                
                draw_caligraphy_line(canvas, prev_x, prev_y, x, y, current_color, brush_size)


            elif current_pattern == "spray":                
                draw_spray_line(canvas, prev_x, prev_y, x, y, current_color, brush_size)


            elif current_pattern == "marker":
                draw_marker_line(canvas, prev_x, prev_y, x, y, current_color, brush_size)

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
        
    # Show brush size text
    cv2.putText(display_frame, f"Brush: {brush_size}px", (30, 410),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    line_color = (20,20,20)

    cv2.line(canvas, (150, 0), (150, FRAME_HEIGHT), line_color, 2)
    cv2.line(canvas, (FRAME_WIDTH-150, 0), (FRAME_WIDTH-150, FRAME_HEIGHT), line_color, 2)

    cv2.line(canvas, (0, 378), (150, 378), line_color, 2)
    cv2.line(canvas, (0, 498), (150, 498), line_color, 2)




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
