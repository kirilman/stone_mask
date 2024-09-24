import json
import base64
import io
from PIL import Image

import torch
from detectron2.model_zoo import get_config
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import os

CONFIG_OPTS = ["MODEL.WEIGHTS", "cascade-mask.pth"]
CONFIDENCE_THRESHOLD = 0.5


def init_context():
    cfg = get_config("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    CONFIG_OPTS.extend(["MODEL.DEVICE", "cpu"])
    cfg.merge_from_list(CONFIG_OPTS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    return predictor


import cv2
import numpy as np


def convert_mask_to_polygon(mask):
    contours = None
    if int(cv2.__version__.split(".")[0]) > 3:
        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )[0]
    else:
        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception(
            "Less then three point have been detected. Can not build a polygon."
        )

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon


model = init_context()
print(dir(model))
print(model.input_format)

pil_image = Image.open("test.jpeg")
image = convert_PIL_to_numpy(pil_image, format="BGR")
predictions = model(image)
instances = predictions["instances"]
pred_boxes = instances.pred_masks
scores = instances.scores
pred_classes = instances.pred_classes
for box, score, label in zip(pred_boxes, scores, pred_classes):
    label = COCO_CATEGORIES[int(label)]["name"]
    mask = box.detach().numpy().astype(np.uint8)

    contours = convert_mask_to_polygon(mask)
    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])
    print(polygon)
