import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def hist_equalize(input):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    hist = histo(input)
    for i in hist:
        print(i)
    return 0


def histo(image):
    hist = np.zeros(256)
    for i in image.ravel():
        hist[i] +=1
        
    return hist

x_axis = range(0,256)

img = cv.imread("./im/low_contrast-man.png")
