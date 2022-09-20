from cProfile import label
from tkinter import N
from turtle import color
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv

def cummu_hist(input):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    hist = histo(input)
    height, width = input.shape[:2]
    pdf_hist = hist/(height*width)
    cdf_hist = np.cumsum(pdf_hist)

    return cdf_hist

def histo(image):
    hist = np.zeros(256)
    for i in image.ravel():
        hist[i] +=1
        
    return hist

def hist_specification(input, input1):
    cdf_hist0 = cummu_hist(input)
    cdf_hist1 = cummu_hist(input1)
    height, width = input.shape[:2]
    new_pixel = np.zeros((height,width))
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    cdf_hist = nearest_cdf(cdf_hist0, cdf_hist1)
    for i in range(height):
        for j in range(width):
            value = cdf_hist[input[i][j]]*255
            new_pixel[i][j] = value
    new_pixel = new_pixel.astype(np.uint8)
    return new_pixel, cdf_hist

def nearest(input_val, input_list1):
    input_list1 = np.asarray(input_list1)
    val = (np.abs(input_val-input_list1)).argmin()
    return input_list1[val]

def nearest_cdf(input_list0, input_list1):
    num = np.zeros(256).astype(np.float32)
    for i, n in enumerate(input_list1):
        num[i] = nearest(n, input_list0)
    return num

x_axis = range(0,256)
img0 = cv.imread("./im/incred_1.png")
img1 = cv.imread("./im/incred_2.png")

new_pixel, cdf_hist = hist_specification(img0, img1)

new_pixel = cv.cvtColor(new_pixel, cv.COLOR_BGR2RGB)
img0 = cv.cvtColor(cv.cvtColor(img0, cv.COLOR_BGR2GRAY), cv.COLOR_BGR2RGB)
img1 = cv.cvtColor(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.COLOR_BGR2RGB)
cdf_hist0 = cummu_hist(img0)
cdf_hist1 = cummu_hist(img1)

plt.subplot(221)
plt.imshow(img0)
plt.title("input image")
plt.subplot(222)
plt.plot(x_axis, cdf_hist0, color="r")
plt.legend(['input image histogram'])
plt.title('Input image histogram')
plt.subplot(223)
plt.imshow(img1)
plt.title("reference image")
plt.subplot(224)
plt.plot(x_axis, cdf_hist1, color="b")
plt.legend(['reference image histogram'])
plt.title('Reference image histogram')
plt.show()

plt.subplot(121)
plt.imshow(new_pixel)
plt.title("Output image")
plt.subplot(122)
plt.plot(x_axis, cdf_hist0, color='r')
plt.plot(x_axis, cdf_hist1, color="b")
plt.plot(x_axis, cdf_hist, color='g')
plt.legend(['input image histogram','reference image histogram','output image histogram'] )
plt.title("SUmmary histogram")
plt.show()

