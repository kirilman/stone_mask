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
import numpy as np

CONFIG_OPTS = ["MODEL.WEIGHTS", "cascade-mask.pth"]
CONFIDENCE_THRESHOLD = 0.5
import cv2


def init_context(context):
    context.logger.info(os.getcwd())
    context.logger.info("Init context...  0%")
    context.logger.info("test run")
    cfg = get_config("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    #    if torch.cuda.is_available():
    #        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cuda'])
    #    else:
    #        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cpu'])
    CONFIG_OPTS.extend(["MODEL.DEVICE", "cpu"])
    cfg.merge_from_list(CONFIG_OPTS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    context.user_data.model_handler = predictor

    context.logger.info("Init context...100%")


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


def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl : ybr + 1, xtl : xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened


def handler(context, event):
    context.logger.info("Run mask-cascade model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.05))
    pil_image = Image.open(buf)
    pil_image.save("test.jpeg")
    image = convert_PIL_to_numpy(pil_image, format="BGR")
    predictions = context.user_data.model_handler(image)

    instances = predictions["instances"]
    pred_boxes = instances.pred_boxes
    pred_masks = instances.pred_masks
    scores = instances.scores
    pred_classes = instances.pred_classes
    results = []
    for mask, score, label in zip(pred_masks, scores, pred_classes):
        label = COCO_CATEGORIES[int(label)]["name"]
        try:
            polygone = convert_mask_to_polygon(mask.detach().numpy().astype(np.uint8))
            polygone = np.array(polygone)
        except:
            continue
        if len(polygone) < 6:
            continue

        if score >= threshold:
            Xmin = int(np.min(polygone[:, 0]))
            Xmax = int(np.max(polygone[:, 0]))
            Ymin = int(np.min(polygone[:, 1]))
            Ymax = int(np.max(polygone[:, 1]))
            cvat_mask = to_cvat_mask(
                (Xmin, Ymin, Xmax, Ymax), mask.detach().numpy().astype(np.uint8)
            )
            results.append(
                {
                    "confidence": str(float(score)),
                    "label": label,
                    "points": polygone.ravel().tolist(),
                    "mask": cvat_mask,
                    "type": "mask",
                }
            )

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
