import time

import cv2
from skimage.measure import compare_ssim as ssim

from src.utils import mse, clean_directory, get_most_often_value
from src.sorts import sort_second, sort_by_blur
from src.constants import MAKING_VIDEO_WITH_DETECTION, \
    MAKING_PHOTO_WITH_DETECTION, \
    INCLUDING_ALL_PHOTOS, \
    MAXIMAL_COMPARE, \
    MINIMAL_MID_SQR_ERR, \
    COUNT_OF_VIDEO_PARTS, \
    COUNT_OF_TOP_PHOTOS
from src.path import RESULT_DIR, VIDEO_STREAM
from src.detection_model import init_model, detect, get_photo_themes
from src.detections_utils import create_video_with_detection
from src.video import get_frame_list

print("[INFO] init model start")

model = init_model()

print("[INFO] init model finish")


def get_the_best_frames_photos(details, main_theme):
    photos = []
    for ind in range(len(details)):
        detail = details[ind]
        theme = detail[0]
        if theme == main_theme or INCLUDING_ALL_PHOTOS:
            photos.append(detail)

    photos.sort(key=sort_second, reverse=True)

    different_photo = get_different_photos(photos)

    return different_photo


def get_different_photos(details):
    current_arr = details
    length = len(current_arr)

    first_photo = details[0]
    next_array = [first_photo]

    for ph_i in range(length - 2):
        current_details = current_arr[ph_i + 1]
        current_photo = current_details[2]
        match_count = 0
        for ch_ph_i in range(len(next_array)):
            next_details = next_array[ch_ph_i]
            next_photo = next_details[2]
            first = cv2.cvtColor(current_photo, cv2.COLOR_BGR2GRAY)
            second = cv2.cvtColor(next_photo, cv2.COLOR_BGR2GRAY)
            sim = ssim(first, second)
            ms = mse(first, second)
            if sim < MAXIMAL_COMPARE and ms > MINIMAL_MID_SQR_ERR:
                match_count = match_count + 1
        if match_count == len(next_array):
            next_array.append(current_details)
            if len(next_array) == COUNT_OF_TOP_PHOTOS:
                break

    return next_array


def get_photos_from_video(file_name):
    print("[INFO] clean directory")

    clean_directory(RESULT_DIR)

    print("[INFO] get frames from video")

    frame_list = get_frame_list(VIDEO_STREAM + file_name)

    print("[INFO] filter frames")

    frame_list.sort(key=sort_by_blur, reverse=True)
    frame_list = frame_list[0:COUNT_OF_VIDEO_PARTS]

    print("[INFO] start detection")

    results = detect(frame_list, model)

    print("[INFO] finish detection")

    photos_details = []
    photos_themes = []

    print("[INFO] handle frames")

    writer = None
    for i in range(len(results)):
        frame = frame_list[i]
        result = results[i]

        [photo_themes, detection] = get_photo_themes(result['class_ids'], result['scores'])
        if photo_themes != "":
            photos_themes.append(photo_themes)
            photos_details.append([photo_themes, detection, frame])

        if MAKING_VIDEO_WITH_DETECTION or MAKING_PHOTO_WITH_DETECTION:
            if MAKING_VIDEO_WITH_DETECTION:
                writer = create_video_with_detection(frame, result, writer)

            if MAKING_PHOTO_WITH_DETECTION:
                cv2.imwrite(RESULT_DIR + str(i) + '_det.jpg', frame)
    if writer:
        writer.release()

    print("[INFO] choose photos")

    video_theme = get_most_often_value(photos_themes)
    the_best_photos = get_the_best_frames_photos(photos_details, video_theme)

    print("[INFO] write photos")

    photo_links = []
    for index in range(len(the_best_photos)):
        photo_data = the_best_photos[index]
        photo = photo_data[2]
        photo_name = str(index) + '_best.jpg'
        photo_link = RESULT_DIR + photo_name
        cv2.imwrite(photo_link, photo)
        ts = time.time()
        photo_links.append('/static/' + photo_name + '?=' + str(ts))

    print("[INFO] finish")

    return photo_links


get_photos_from_video('cats.mp4')
