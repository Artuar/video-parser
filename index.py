import os
import sys
import numpy as np
import cv2
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from statistics import mode, StatisticsError

MAKING_VIDEO_WITH_DETECTION = False
MAKING_PHOTO_WITH_DETECTION = False
MAKING_PHOTO_WITHOUT_DETECTION = False
INCLUDING_ALL_PHOTOS = True
MINIMAL_DETECTION = 0.95

COUNT_OF_VIDEO_PARTS = 30
COUNT_OF_TOP_PHOTOS = 5

FILE_NAME = "cats.mp4"
VIDEO_STREAM = "./videos/"

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
RESULT_DIR = os.path.join(ROOT_DIR, "result") + '/'
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Special settings for making video
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# clean directory
file_list = [f for f in os.listdir(RESULT_DIR)]
for f in file_list:
    os.remove(os.path.join(RESULT_DIR, f))


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # NUM_CLASSES = 37  # COCO has 80 classes
    NUM_CLASSES = 81


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = COUNT_OF_VIDEO_PARTS
    # MAX_GT_INSTANCES = 100
    # TRAIN_ROIS_PER_IMAGE = 50
    # BACKBONE = "resnet50" #not working at all!
    # RPN_ANCHOR_STRIDE = 2
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 500
    IMAGE_MIN_DIM = 400  # really much faster but bad results
    IMAGE_MAX_DIM = 512
    # DETECTION_MAX_INSTANCES = 50 #a little faster but some instances not recognized


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = visualize.random_colors(n_instances)
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = visualize.apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
    return image


def get_theme_by_class(cl):
    if cl > 80:
        return 'other'
    if cl > 57:
        return 'home'
    if cl > 41:
        return 'food/restaurant'
    if cl > 24:
        return 'sport/walking'
    if cl > 13:
        return 'animals'
    if cl > 1:
        return 'travel/transport'
    if cl > 0:
        return 'people'
    return 'other'


def get_photo_themes(classes, detections):
    if len(detections) == 0:
        return ['other', 0]
    lst = detections.tolist()
    mid_detect = np.array(lst).mean()
    if mid_detect < MINIMAL_DETECTION:
        return ['other', 0]

    max_detect = np.array(lst).max()
    index_of_max = lst.index(max_detect)
    return [get_theme_by_class(classes[index_of_max]), max_detect]


def sort_second(val):
    return val[1]


def sort_by_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def get_the_best_frames_photos(details, main_theme):
    photos = []
    for ind in range(len(details)):
        detail = details[ind]
        theme = detail[0]
        if theme == main_theme or INCLUDING_ALL_PHOTOS:
            photos.append(detail)

    photos.sort(key=sort_second, reverse=True)

    return photos


def get_most_often_value(arr):
    try:
        return mode(arr)
    except StatisticsError:
        return arr[0] or 'other'


config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person',

               'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter',

               'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe',

               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
               'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket',

               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake',

               'chair', 'couch', 'potted plant', 'bed', 'dining table',
               'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

photos_details = []
photos_themes = []
video_theme = ''

# Initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(VIDEO_STREAM + FILE_NAME)
count_of_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
writer = None
inde = 0
frame_list = []
while inde < COUNT_OF_VIDEO_PARTS*2:
    # go to next n frames forward
    frame_number = inde * count_of_frames / (COUNT_OF_VIDEO_PARTS*2)
    vs.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    inde += 1

    if MAKING_PHOTO_WITHOUT_DETECTION:
        cv2.imwrite(RESULT_DIR + str(inde) + '.jpg', frame)

    # If the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        print("Not grabbed.")
        break

    # Run detection
    # reduced_frame = cv2.resize(frame, (480, 260))
    # blurred_frame = cv2.blur(reduced_frame, (5, 5))
    # cv2.imwrite(RESULT_DIR + str(i) + '_reduced.jpeg', blurred_frame)
    frame_list.append(frame)

frame_list.sort(key=sort_by_blur, reverse=True)
print(len(frame_list))
frame_list = frame_list[0:COUNT_OF_VIDEO_PARTS]
print(len(frame_list))

results = model.detect(frame_list, verbose=1)

for i in range(len(results)):
    frame = frame_list[i]

    # Visualize results
    r = results[i]

    [photo_themes, detection] = get_photo_themes(r['class_ids'], r['scores'])
    if photo_themes != "":
        photos_themes.append(photo_themes)
        photos_details.append([photo_themes, detection, frame])
        # photos_details.append([photo_themes, max_detection, frame])

    if MAKING_VIDEO_WITH_DETECTION or MAKING_PHOTO_WITH_DETECTION:
        masked_frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                         class_names, r['scores'])

        if MAKING_VIDEO_WITH_DETECTION:
            # Check if the video writer is None
            if writer is None:
                # Initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(RESULT_DIR + FILE_NAME, fourcc, 30,  # TODO it depends on COUNT_OF_VIDEO_PARTS
                                         (masked_frame.shape[1], masked_frame.shape[0]), True)
            # Write the output frame to disk
            writer.write(masked_frame)

        if MAKING_PHOTO_WITH_DETECTION:
            cv2.imwrite(RESULT_DIR + str(i) + '_rec.jpg', frame)

video_theme = get_most_often_value(photos_themes)
the_best_photos = get_the_best_frames_photos(photos_details, video_theme)

for index in range(COUNT_OF_TOP_PHOTOS):  # TODO but no more then len(the_best_photos)
    photo_data = the_best_photos[index]
    photo = photo_data[2]
    cv2.imwrite(RESULT_DIR + str(index) + '_best.jpg', photo)
    print(photo_data[1], photo_data[0])

# Release the file pointers
print("[INFO] cleaning up...")
if writer:
    writer.release()
