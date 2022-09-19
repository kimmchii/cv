import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread("./im/low_contrast-man.png")
def thresholds(input, cutoff):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    height, width = input.shape[:2]
    new_pixel = np.zeros((height,width)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            if input[i][j] >= cutoff:
                new_pixel[i][j] = 255
            else:
                new_pixel[i][j] = 0
    
    img = cv.cvtColor(input, cv.COLOR_GRAY2RGB)
    new_p = cv.cvtColor(new_pixel, cv.COLOR_GRAY2RGB)
    return new_p

def histogram(input, input2):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([input],[0],None,[256],[0,256])
    plot_img = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    input2 = cv.cvtColor(input2, cv.COLOR_BGR2GRAY)
    hist2 = cv.calcHist([input2],[0],None,[256],[0,256])
    plot_img2 = cv.cvtColor(input2, cv.COLOR_BGR2RGB)
    
    plt.subplot(221)
    plt.imshow(plot_img)
    plt.title("Original image")
    plt.subplot(222)
    plt.plot(hist)
    plt.title("Original image histogram")
    plt.subplot(223)
    plt.imshow(plot_img2)
    plt.title("Converted image")
    plt.subplot(224)
    plt.plot(hist2)
    plt.title("Converted image histogram")
    plt.show()
    

result = thresholds(img, 210)
histogram(img, result)
