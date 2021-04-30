
https://user-images.githubusercontent.com/73319539/116725706-3873a600-a9da-11eb-8d18-04d2c89e4c25.mp4

import cv2 as cv
import numpy as np
import argparse
#Enter the path of the image that needs to be converted
imgpath = input("input img path:")
#Load pictures
img = cv.imread(imgpath)
#Enter the path of the algorithm model
model = input("input model name:")
#Get the width and height of the picture
(inHeight, inWidth) = img.shape[:2]
#Perform image preprocessing
inp = cv.dnn.blobFromImage(img, 1.0, (inWidth, inHeight),(103.939, 116.779, 123.68), swapRB=False, crop=False)
#Loading algorithm model
net = cv.dnn.readNetFromTorch(model)
#Execute through algorithm model Preprocessed data inp
net.setInput(inp)
#Get the results after the algorithm is executed
out = net.forward()
#Conversion data
out = out.reshape(3, out.shape[2], out.shape[3])
out[0] += 103.939
out[1] += 116.779
out[2] += 123.68
#Convert to data that can be saved as a picture
out = out.transpose(1, 2, 0)
#Enter the name of the saved data
outputname = input("input your output file name:")
#Save the results after the algorithm runs
cv.imwrite(outputname, out, [20, 80])
![image](https://user-images.githubusercontent.com/73319539/116725750-488b8580-a9da-11eb-92be-1a285c7cfa42.png)
# week-7
