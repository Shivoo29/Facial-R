# Real-Time Face Matching

This project implements real-time face matching using webcam input. It compares the faces detected in the webcam feed with a reference image to determine if there's a match.

## Features

- Real-time face detection and matching
- Uses webcam input
- Compares detected faces with a reference image
- Provides visual feedback on match status

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- DeepFace
- A webcam


## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Shivoo29/Facial-R.git
    cd Facial-R/
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python-headless deepface
    ```

3. Place a reference image named `reference.jpg` in the project directory.

## Usage

1. Run the script:
    ```bash
    python main.py
    ```

2. The webcam feed will open in a new window.

3. The script will compare faces in the webcam feed with the reference image every 30 frames.

4. The window will display "MATCH!!" in green if a match is found, or "NO MATCH!!" in red otherwise.

5. Press 'q' to quit the application.

## How it works

- The script captures video from the default webcam.
- Every 30 frames, it takes a snapshot and compares it with the reference image using DeepFace.
- The comparison is done in a separate thread to avoid blocking the main loop.
- The result is displayed on the video feed in real-time.

## Customization

- You can change the reference image by replacing `reference.jpg` with your own image.
- Adjust the frame skip rate by modifying the `counter % 30` condition in the main loop.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/Shivoo29/Facial-R/issues) if you want to contribute.

## Author

SKJ
