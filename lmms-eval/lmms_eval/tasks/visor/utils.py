from collections import defaultdict
import os
import datetime
import json
# from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path
import yaml
import sys, string
from typing import List, Dict, Optional, Union
import re
import PIL
import numpy as np
from loguru import logger as eval_logger

import io
from petrel_client.client import Client
from collections import defaultdict
import os
import datetime
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path
import yaml
import sys, string
from typing import List, Dict, Optional, Union
import re
import PIL
import numpy as np
from loguru import logger as eval_logger

import io
from petrel_client.client import Client
import json
import os
from io import BytesIO

import requests
from loguru import logger as eval_logger
from PIL import Image
import re


def visor_doc_to_visual(doc, lmms_eval_specific_kwargs=None):


    image_path = doc['path']
    image = Image.open(image_path)
    
    return [image.convert("RGB")]

def visor_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    object = doc['object']
    question = f"Bounding box coordinates are specified in the format [x min, y min, x max, y max]. All values are floating point numbers bounded between 0 and 1. Please provide the bounding box coordinate of the {object}."

    return question

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def visor_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """

    pred = results[0]
    # type = doc["Question_Type"]
    gt_answer = doc['gt']

    pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"
    match = re.search(pattern, pred)
    if match:
        pred_box = [float(match.group(i)) for i in range(1, 5)]
        score = compute_iou(pred_box,gt_answer)
    else:
        score = 0
    
    # Use re.search to find the first match of the pattern in the input string
    data_dict = {"pred_answer": pred, "gt_answer": gt_answer, "score": score}

    return {"visor_iou": data_dict}

def visor_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_answered = 0
    total_correct = 0
    for result in results:
        if result["pred_answer"] != "":
            total_answered += 1
            total_correct += result["score"]

    return 100 * total_correct / total_answered if total_answered > 0 else 0