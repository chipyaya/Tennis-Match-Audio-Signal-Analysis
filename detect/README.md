# WN-FPJ-UTCS
## Human detection
Download the pretrained yolov3 weight to detect/
```
wget https://pjreddie.com/media/files/yolov3.weights
```
Usage:
```
python3 run.py --directory=../video2image/
```
This will generate a txt file for every image. Every line in the txt file is a detected human instance with four variables, which are the x, y, width, height for the bounding box.