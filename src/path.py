import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN

RESULT_DIR = os.path.join(ROOT_DIR, "static") + '/'
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

VIDEO_STREAM = "./videos/"

sys.path.append(ROOT_DIR)  # To find local version of the library
