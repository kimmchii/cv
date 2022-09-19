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
    
x_axis = range(0,256)
img0 = cv.imread("./im/incred_1.png")
img1 = cv.imread("./im/incred_3.png")

hist0 = histo(img0)
hist1 = histo(img1)
plt.subplot(121)
plt.plot(x_axis, hist0 )
plt.subplot(122)
plt.plot(x_axis, hist1)
plt.show()