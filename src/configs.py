from mrcnn.config import Config
from src.constants import COUNT_OF_VIDEO_PARTS


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

