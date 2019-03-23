import os
import sys
import numpy as np
import cv2
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

MAKING_VIDEO_WITH_DETECTION = False
MAKING_PHOTO_WITH_DETECTION = False
MAKING_PHOTO_WITHOUT_DETECTION = True

COUNT_OF_VIDEO_PARTS = 5
COUNT_OF_TOP_PHOTOS = 5

FILE_NAME = "cats.mp4"

VIDEO_STREAM = "./videos/"
PHOTOS_WITH_DETECTION_DIR_OUT = "./result/photos_with_detection/"
PHOTOS_DIR_OUT = "./result/photos/"
VIDEO_STREAM_OUT = "./result/"

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Special settings for making video
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    IMAGES_PER_GPU = 1
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


def get_the_best_frames_indexes(details):
    indexes = []
    return indexes


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

type_of_classes = [
    ['people', 1, 1],
    ['travel/transport', 2, 13],
    ['animals', 14, 24],
    ['sport/walking', 25, 40],
    ['food/restaurant', 41, 57],
    ['home', 58, 81],
]

photos_details = []

# Initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(VIDEO_STREAM + FILE_NAME)
count_of_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
writer = None
i = 0
while i < COUNT_OF_VIDEO_PARTS:
    # go to next n frames forward
    frame_number = i * count_of_frames / COUNT_OF_VIDEO_PARTS
    vs.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    i += 1

    if MAKING_PHOTO_WITHOUT_DETECTION:
        cv2.imwrite(PHOTOS_DIR_OUT + str(i) + '.jpeg', frame)

    # If the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        print("Not grabbed.")
        break

    # Run detection
    results = model.detect([frame], verbose=1)
    # Visualize results
    r = results[0]

    photos_details.append([r['class_ids'], r['scores'], frame_number])

    if MAKING_VIDEO_WITH_DETECTION | MAKING_PHOTO_WITH_DETECTION:
        masked_frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                         class_names, r['scores'])

        if MAKING_VIDEO_WITH_DETECTION:
            # Check if the video writer is None
            if writer is None:
                # Initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(VIDEO_STREAM_OUT, fourcc, 30,
                                         (masked_frame.shape[1], masked_frame.shape[0]), True)
            # Write the output frame to disk
            writer.write(masked_frame)

        if MAKING_PHOTO_WITH_DETECTION:
            cv2.imwrite(PHOTOS_WITH_DETECTION_DIR_OUT + str(i) + '_rec.jpeg', frame)

the_best_frames = get_the_best_frames_indexes(photos_details)

# Release the file pointers
print("[INFO] cleaning up...")
if writer:
    writer.release()
