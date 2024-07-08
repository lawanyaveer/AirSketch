# AirSketch: Real-time Hand Gesture-based Drawing Application 🎨✋

## Overview 📖

AirSketch is an advanced computer vision application that enables users to draw in the air using hand gestures. It utilizes real-time hand tracking and gesture recognition to create a virtual drawing canvas. This project demonstrates the integration of computer vision techniques, gesture recognition algorithms, and real-time video processing.

## Technical Stack 🛠️

- **Python 3.7+**: Core programming language 🐍
- **OpenCV 4.5+**: Computer vision library for image processing and drawing 📷
- **MediaPipe 0.8.9+**: Machine learning framework for hand tracking ✋
- **NumPy 1.19+**: Numerical computing library for efficient array operations 📊

## Key Features 🌟

1. **Real-time Hand Tracking**: Utilizes MediaPipe's hand landmark detection model to track 21 3D hand landmarks at 30+ FPS.
2. **Gesture Recognition**: Implements custom logic to detect raised index finger for drawing initiation and termination.
3. **Dynamic Color Selection**: Provides an interactive UI for real-time color switching during drawing.
4. **Adaptive Smoothing**: Employs a distance-based point sampling technique to reduce jitter and improve line quality.
5. **Performance Optimization**: Incorporates frame resolution reduction and efficient drawing algorithms to minimize latency.
6. **Multi-layer Rendering**: Combines the input video feed with the drawing canvas using alpha blending for a seamless user experience.

## System Architecture 🏗️

The application follows a modular architecture:

1. **Input Module**: Captures and preprocesses video frames from the webcam.
2. **Hand Detection Module**: Utilizes MediaPipe to detect and track hand landmarks.
3. **Gesture Recognition Module**: Analyzes hand landmark positions to recognize drawing gestures.
4. **Drawing Module**: Manages the canvas state and renders lines based on recognized gestures.
5. **UI Module**: Handles the creation and interaction with the color selection and clear button interface.
6. **Output Module**: Combines processed frames, UI elements, and the drawing canvas for final display.

## Key Algorithms 🔍

### Hand Landmark Detection

Utilizes MediaPipe's palm detection model followed by a hand landmark model to identify 21 3D landmarks of a hand.

### Index Finger Raise Detection

```python
def is_index_finger_raised(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
```

This function compares the y-coordinates of the index finger tip (landmark 8) and the middle knuckle (landmark 6) to determine if the index finger is raised.

### Adaptive Line Drawing

```python
if prev_point and np.linalg.norm(np.array(index_finger_tip) - np.array(prev_point)) > min_distance:
    cv2.line(canvas, prev_point, index_finger_tip, colors[colorIndex], line_thickness)
    prev_point = index_finger_tip
```

This algorithm ensures smooth line drawing by only rendering lines when the finger has moved a significant distance, reducing jitter and improving performance.

## Performance Considerations ⚡

1. **Frame Resolution**: Reduced to 640x480 to balance between processing speed and visual quality.
2. **Hand Detection Confidence**: Set to 0.5 to optimize the trade-off between accuracy and speed.
3. **Drawing Optimization**: Direct line drawing instead of complex smoothing algorithms to reduce latency.
4. **UI Rendering**: Pre-rendered UI elements to minimize per-frame computation.

## Installation and Dependencies 🚀

1. Ensure Python 3.7+ is installed.
2. Install required libraries:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AirSketch.git
   cd Air_Sketch
   ```

## Usage 📋

Run the application:

```bash
python Air_Sketch.py
```

- Use your index finger to draw in the air. ✍️
- Touch the color buttons at the top of the screen to change the drawing color. 🎨
- Use the 'CLEAR' button to reset the canvas. 🔄
- Press 's' to save your drawing, '+'/'-' to adjust line thickness, and 'q' to quit.

## Future Enhancements 🔮

1. Implement multi-hand support for collaborative drawing.
2. Integrate machine learning for gesture customization.
3. Develop 3D drawing capabilities using depth estimation techniques.
4. Optimize for mobile devices using TensorFlow Lite.

## Contributing 🤝

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments 🙏

- MediaPipe team for their hand-tracking solution
- OpenCV contributors for the comprehensive computer vision toolkit