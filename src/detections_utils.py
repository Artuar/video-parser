from mrcnn import visualize
import numpy as np
import cv2

from src.detection_model import class_names
from src.path import RESULT_DIR


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


def create_video_with_detection(frame, result, writer):
    masked_frame = display_instances(frame, result['rois'], result['masks'], result['class_ids'],
                                     class_names, result['scores'])

    # Check if the video writer is None
    if writer is None:
        # Initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(RESULT_DIR + "res.mp4", fourcc, 30,  # TODO it depends on COUNT_OF_VIDEO_PARTS
                                 (masked_frame.shape[1], masked_frame.shape[0]), True)
    # Write the output frame to disk
    writer.write(masked_frame)

    return writer
