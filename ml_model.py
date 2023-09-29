# %%
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

__author__ = "ALI FALAHATI"


images_path = Path("data/images")
        


def predict(image_ids):
    images = []
    for img in image_ids:
        images.append(str(images_path/img))

    model = YOLO('yolov8n.pt')
    result = model.predict(images, classes = [2])

    output = []
    i = 0
    for res in result:
        if len(res.boxes.cls) != 0:
            for j in range(len(res.boxes.cls)):
                conf = res.boxes.conf[j].item()
                x = res.boxes.xyxy[j].numpy()[0]
                y = res.boxes.xyxy[j].numpy()[1]

                dic = {
                    'id' : image_ids[i],
                    'x' : x,
                    'y' : y,
                    'conf' : conf
                }

                output.append(dic)
        else:
            dic = {
                'id' : image_ids[i],
                'x' : None,
                'y' : None,
                'conf' : None
            }

            output.append(dic)

        i += 1
    return output
