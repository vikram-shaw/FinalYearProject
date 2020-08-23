#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:14:33 2020

@author: yes_v
"""


import cv2
from skimage import measure
import numpy as np
from imutils import contours
import imutils

def median_filter(img):
    img_out = img.copy()
    
    height = img.shape[0]
    width = img.shape[1]
    for i in np.arange(2, height-2):
    	for j in np.arange(2, width-2):
    		neighbors = []
    		for k in np.arange(-2, 2):
    			for l in np.arange(-2, 2):
    				a = img.item(i+k,j+l)
    				neighbors.append(a)
    		neighbors.sort()
    		median = neighbors[8]
    		b = median
    		img_out.itemset((i,j), b)
    return img_out
    

original = cv2.imread('/home/yes_v/Downloads/kidney.png',1)

cv2.imshow('original',original)
gray = cv2.cvtColor(original,cv2.COLOR_BGRA2GRAY)
cv2.imwrite('gray.jpg',gray);
blurred = cv2.GaussianBlur(gray,(5,5),1)

cv2.imshow('blurred',blurred)
cv2.imwrite('gaussian.jpg',blurred);

#kernel = np.ones((3,3),np.float32)/9

#blurred = cv2.filter2D(gray, -1, kernel)

#blurred = median_filter(gray)

thresh = cv2.threshold(blurred, 198, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('threshold.jpg',thresh);

cv2.imshow('thresholded',thresh)
thresh = cv2.erode(thresh, None, iterations = 1)

cv2.imshow('erossion',thresh)
cv2.imwrite('erode.jpg',thresh);
thresh = cv2.dilate(thresh, None, iterations = 1)

cv2.imshow('dilated',thresh)
cv2.imwrite('dilate.jpg',thresh);

labels = measure.label(thresh, background=0)
mask = np.zeros(thresh.shape, dtype='uint8')

stone = False

for label in np.unique(labels):
    if label == 0:
        continue
    labelMask = np.zeros(thresh.shape,dtype='uint8')
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    
    if numPixels > 10 and numPixels < 300:
        mask = cv2.add(mask, labelMask)
        stone = True

if(stone):
    cv2.putText(original,"STONE DETECTED",(original.shape[0]//2,10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
    print("stone is detected")
else:
    cv2.putText(original,"STONE NOT DETECTED",(original.shape[0]//2,10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
    print("stone not detected")
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts =  imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

for (i,c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(original,(int(cX),int(cY)),int(radius+5),(225,225,0),2)
cv2.imwrite('detected_image.jpg',original);

cv2.imshow('result',original)
cv2.waitKey(0)
cv2.destroyAllWindows()