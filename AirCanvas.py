import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set width and height of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Could not access the camera.")
else:
    print("Camera is working!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Mirror the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Display the frame
    cv2.imshow("My Air Drawing App", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()