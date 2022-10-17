
from hashlib import new
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def hist_equalize(input):
    cdf_hist = cummu_hist(input)
    height, width = input.shape[:2]
    new_pixel = np.zeros((height,width))
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    for i in range(height):
        for j in range(width):
            value = cdf_hist[input[i][j]]*255
            new_pixel[i][j] = value
    new_pixel = new_pixel.astype(np.uint8)
    new_pixel = cv.cvtColor(new_pixel, cv.COLOR_BGR2RGB)
    return new_pixel, cdf_hist

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

x_axis = range(0,256)

#img = cv.imread("./im/low_contrast-man.png")

img = cv.imread("./im/aka.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.cvtColor(gray, cv.COLOR_BGR2RGB)
cdf_hist = cummu_hist(img)
result,  cdf = hist_equalize(img)
new_cdf = cummu_hist(result)


plt.subplot(221)
plt.imshow(gray)
plt.title("Input image")
plt.subplot(222)
plt.plot(x_axis, cdf_hist)
plt.title("Input image histogram")
plt.subplot(223)
plt.imshow(result)
plt.title("Output image")
plt.subplot(224)
plt.plot(x_axis, new_cdf)
plt.title("Output image histogram")
plt.show()
#result = hist_equalize(img)
