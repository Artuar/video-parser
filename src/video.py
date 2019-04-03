import cv2
import os
from src.constants import COUNT_OF_VIDEO_PARTS, MAKING_PHOTO_WITHOUT_DETECTION, EXTRA_FRAMES_MULTIPLY
from src.path import RESULT_DIR

# Special settings for making video
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_frame_list(url):
    # Initialize the video stream and pointer to output video file
    vs = cv2.VideoCapture(url)
    count_of_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    frame_list = []

    while frame_index < COUNT_OF_VIDEO_PARTS * EXTRA_FRAMES_MULTIPLY:
        # go to next n frames forward
        frame_number = frame_index * count_of_frames / (COUNT_OF_VIDEO_PARTS * EXTRA_FRAMES_MULTIPLY)
        vs.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        frame_index += 1

        if MAKING_PHOTO_WITHOUT_DETECTION:
            cv2.imwrite(RESULT_DIR + str(frame_index) + '.jpg', frame)

        # If the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            print("Not grabbed.")
            break

        frame_list.append(frame)

    return frame_list
