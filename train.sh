#!/bin/bash

source venv/activate/bin

yolo detect train data=dataset/ball/data.yaml model=yolo11s.pt epochs=100 imgsz=720 batch=48 rect=False mosaic=0.0 fliplr=0.0 scale=0.0 hsv_s=0.0 hsv_h=0.0 erasing=0.0 patience=20 cache=disk
