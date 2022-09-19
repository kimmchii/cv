from select import POLLOUT
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

def histogram(input):
    hist = cv.calcHist([input],[0],None,[256],[0,256])
    plot_img = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    
    mx, mn = finding_pixel(plot_img)
    print(mx)
    print(mn)
    plt.subplot(121)
    plt.imshow(plot_img)
    plt.subplot(122)
    plt.plot(hist)
    plt.show()


def finding_pixel(input):
    plot_img = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    mx = max(plot_img.ravel())
    mn = min(plot_img.ravel())
    return mx, mn


def CS(input, low, high):
    mx, mn = finding_pixel(input)
    height, width = input.shape[:2]
    new_pix = np.zeros((height,width))
    # low = minimum value of the output pixel
    # high = maximum value of the output pixel
    # mn = minimum value of the input pixel
    # mx = maximum value of the input pixel
    for i in range(height):
        for j in range(width):
            new_pix[i][j] = low + ((input[i][j]-mn)*(high-low))/(mx-mn)
    
   
    input = cv.cvtColor(input, cv.COLOR_GRAY2RGB)
    new_pix = new_pix.astype(np.uint8)
    new_pix = cv.cvtColor(new_pix, cv.COLOR_GRAY2RGB)
    plt.subplot(121)
    plt.imshow(input)
    plt.title("original")
    plt.subplot(122)
    plt.imshow(new_pix)
    plt.title("converted")
    plt.show()
  
    

img = cv.cvtColor(cv.imread("./im/low_contrast-man.png"), cv.COLOR_BGR2GRAY)

#histogram(img)
CS(img, 100, 255)
