import os
import mrcnn.model as modellib
from src.configs import InferenceConfig
from mrcnn import utils
import numpy as np
import imutils

from src.path import COCO_MODEL_PATH, MODEL_DIR
from src.constants import MINIMAL_DETECTION

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
    return [get_theme_by_class(classes[index_of_max]), mid_detect]


def download_trained():
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)


def init_model():
    download_trained()

    config = InferenceConfig()
    # config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


def detect(photos, model):
    light_photos = []
    for photo in photos:
        light_photos.append(imutils.resize(photo, width=450))

    return model.detect(light_photos)


