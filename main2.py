## for those who don't have any camera with there pc
import threading
import cv2
from deepface import DeepFace

# URL of the DroidCam video feed
video_url = "http://192.168.0.109:4343/video"

# Open the video feed from DroidCam
cap = cv2.VideoCapture(video_url)

# Set the resolution of the video feed (if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables
counter = 0
face_match = False

# Load the reference image
reference_img = cv2.imread("reference.jpg")

def check_face(frame):
    global face_match
    try:
        # Use DeepFace to verify the face in the frame against the reference image
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        # Check the face every 30 frames
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass

        counter += 1

        # Display match status on the frame
        if face_match:
            cv2.putText(frame, "MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Show the frame
        cv2.imshow("video", frame)

    # Exit if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video feed and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
