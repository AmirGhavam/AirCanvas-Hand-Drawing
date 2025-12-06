# 🎨 AirCanvas

**Author:** Amir Ghavam

**AirCanvas** is a computer vision-based application that allows users to draw in the air using hand gestures. Built with **Python**, **OpenCV**, and **MediaPipe**, this project creates a virtual canvas where your index finger becomes the brush.

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Solutions-orange)

<!-- You can add a screenshot or gif here later -->
<!-- ![AirCanvas Demo](path/to/your/screenshot.png) -->

## ✨ Features

*   **Touchless Drawing:** Tracks the index finger to draw on the screen in real-time.
*   **Multiple Brush Styles:**
    *   **Normal:** Standard solid line.
    *   **Neon:** Glowing, multi-layered light effect.
    *   **Calligraphy:** Dynamic thickness based on stroke direction.
    *   **Spray:** Particle-based airbrush effect.
    *   **Marker:** Semi-transparent, layered ink effect.
*   **Gesture Control:**
    *   **Draw:** Raise **only** your index finger.
    *   **Hover/Stop Drawing:** Raise two or more fingers (or close your hand).
*   **Interactive UI:**
    *   Color palette selection (Blue, Red, Green, White, Yellow, Pink).
    *   Brush size adjustment (+ / -).
    *   Eraser tool.
*   **Utility Functions:**
    *   **Clear:** Wipes the canvas clean.
    *   **Save:** Saves just the drawing as a PNG.
    *   **Screenshot:** Saves the drawing combined with the webcam feed.
    *   **Toggle Cam:** Hides the webcam feed to show only the canvas (black background).
*   **Smooth Tracking:** Implements a point buffer system to reduce jitter and create smooth strokes.

## 🛠️ Dependencies

To run this project, you need to have Python installed along with the following libraries:

*   `opencv-python`
*   `mediapipe`
*   `numpy`

## 📥 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AmirGhavam/AirCanvas-Hand-Drawing.git
    cd AirCanvas
    ```

2.  **Install the required packages:**
    ```bash
    pip install opencv-python mediapipe numpy
    ```

## 🚀 Usage

1.  **Run the script:**
    ```bash
    python AirCanvas.py
    ```
2.  **Webcam Setup:** Ensure your webcam is connected. The application defaults to the primary camera (`index 0`) and attempts to set the resolution to 1920x1080.
3.  **How to Draw:**
    *   Stand in front of the camera.
    *   Raise your **Index Finger** to start drawing.
    *   The application mirrors your movement for natural interaction.
4.  **Using the Menu:**
    *   Move your finger over the buttons on the **Left** (Colors/Brushes) or **Right** (Actions).
    *   The buttons will "pop out" visually when hovered.
    *   The selection is activated when your fingertip enters the button area.
5.  **Exit:**
    *   Hover over the **EXIT** button, or press `q` on your keyboard.

## 🎮 Controls & Interface

| **Left Sidebar** | **Function** |
| :--- | :--- |
| **Colors** | Select Blue, Red, Green, White, Yellow, or Pink ink. |
| **- / +** | Decrease or Increase the brush size. |
| **Eraser** | Switches to a black brush to erase content. |
| **Neon** | Switches to the Neon glowing brush pattern. |
| **Calligraphy** | Switches to the Calligraphy brush pattern. |
| **Spray** | Switches to the Spray paint pattern. |
| **Marker** | Switches to the Marker pen pattern. |

| **Right Sidebar** | **Function** |
| :--- | :--- |
| **Clear** | Deletes everything on the canvas. |
| **Save** | Saves `drawing.png` (Canvas only). |
| **Screenshot** | Saves `screenshot.png` (Canvas + Webcam). |
| **ToggleCam** | Turns the video feed on/off (focus on drawing). |
| **Exit** | Closes the application. |

## 🧠 How It Works

1.  **Hand Detection:** Uses `mediapipe` to detect hand landmarks.
2.  **Finger Logic:** The script checks which fingers are raised. Drawing mode is enabled *only* when the index finger is up and the middle finger is down.
3.  **Coordinates:** Converts the normalized coordinates (0.0 to 1.0) provided by MediaPipe into pixel coordinates (1920x1080).
4.  **Smoothing:** A `point_buffer` stores the last few detected positions and averages them to prevent shaky lines caused by webcam noise.
5.  **Canvas Blending:** The drawing exists on a black NumPy array (`canvas`) which is blended with the live video feed (`frame`) using `cv2.addWeighted`.

## 🔮 Future Improvements

*   Add support for gesture-based undo (e.g., showing a thumb-down gesture).
*   Implement a color picker wheel for custom colors.
*   Add support for multi-hand tracking (drawing with two hands).

## 📄 License

This project is open-source and available for use.
