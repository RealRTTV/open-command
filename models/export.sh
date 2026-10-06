#!/bin/bash

yolo export model="$1" format=engine imgsz=704 batch=16 simplify=False workspace=4 quantize=16
