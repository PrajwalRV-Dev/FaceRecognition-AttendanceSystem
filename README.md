# Face Recognition Attendance System with Liveness Detection

This project is an advanced attendance system that uses facial recognition to identify individuals and mark their attendance. To prevent spoofing attacks from photos or videos, it incorporates a robust, two-step liveness detection process.

## Features

*   **Real-time Face Recognition**: Identifies known individuals from a live webcam feed.
*   **Dynamic Face Enrollment**: Simply add a person's image to the `known_faces` directory to enroll them.
*   **Advanced Liveness Detection**: Ensures that the person is physically present and not a static image or video replay.
    *   **Blink Detection**: The system first verifies that the person is blinking naturally.
    *   **Challenge-Response Head Movement**: After a successful blink, the system issues a random challenge (e.g., "Turn head to the left"), requiring a physical response.
*   **Attendance Logging**: Records the name, date, and time of each verified attendance into an `attendance.csv` file, marking each person only once per day.

## How it Works

1.  **Load Known Faces**: The application starts by loading face encodings from images stored in the `known_faces` directory.
2.  **Launch Webcam**: It then accesses the primary webcam for real-time video capture.
3.  **Detect and Recognize**: In each frame, the system detects faces and compares them against the known faces.
4.  **Verify Liveness**: For a recognized person, it initiates a two-step liveness check:
    1.  It waits for the person to blink.
    2.  It then prompts them to turn their head left or right.
5.  **Mark Attendance**: Once liveness is confirmed, the system logs the person's attendance in `attendance.csv`.
6.  **Visual Feedback**: The video feed provides real-time status updates, displaying the person's name and the current liveness check prompt.

## Setup and Installation

1.  **Clone the repository or download the source code.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .\\venv\\Scripts\\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required libraries:**
    You will need `dlib`, which can sometimes be tricky to install. It's often easier to install it via a pre-compiled wheel if you run into issues with a direct `pip install`.
    ```bash
    pip install opencv-python numpy face-recognition pandas
    ```

    **Note on `dlib` installation:** If `pip install face-recognition` fails, you may need to install `CMake` and a C++ compiler first. Alternatively, you can search for a pre-compiled `dlib` wheel (`.whl` file) that matches your Python version and system architecture.

## How to Use

1.  **Enroll Faces**:
    *   Create a directory named `known_faces` in the project's root folder if it doesn't exist.
    *   Add `.jpg` or `.png` images of the people you want to recognize into this directory.
    *   Name the image files with the person's name (e.g., `john_doe.jpg`). The system will use the filename as their name in the attendance log.

2.  **Run the application**:
    Execute the main script from your terminal:
    ```bash
    python face_attendance.py
    ```

3.  **Interacting with the System**:
    Once the application is running, a window will appear with the webcam feed. Here's what to do:
    *   **Position Your Face**: Make sure your face is well-lit and centered in the frame.
    *   **Blink to Verify**: The system will first display the text "Blink to verify". Simply look at the camera and blink naturally.
    *   **Perform the Head-Turn Challenge**: After a successful blink, the system will challenge you to prove you are live by displaying either "Turn head to the left" or "Turn head to the right". Turn your head in the requested direction.
    *   **Get Confirmation**: Once you correctly turn your head, the status will change to "Liveness Verified", and your attendance for the day will be logged in `attendance.csv`. If you were already marked present, it will show "Attendance Marked".

4.  **Quit the application**:
    Press the `q` key with the video window in focus to stop the program.

5.  **Check the Attendance Log**:
    An `attendance.csv` file will be created or updated in the project directory with the attendance records.

## File Structure

```
.
├── face_attendance.py      # Main application script
├── known_faces/            # Directory to store images of known people
│   ├── person1.jpg
│   └── person2.png
├── attendance.csv          # Log file for attendance (created on first run)
└── README.md               # This file
``` 