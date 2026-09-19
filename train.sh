#!/bin/bash

yolo detect train data=dataset/ball/data.yaml model=yolo11s.pt epochs=100 imgsz=704 batch=16 rect=False mosaic=0.0 fliplr=0.0 scale=0.0 hsv_s=0.0 hsv_h=0.0 erasing=0.0 patience=20 cache=disk
