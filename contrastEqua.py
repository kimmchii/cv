import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

def histogram(input, input2):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([input],[0],None,[256],[0,256])
    plot_img = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    input2 = cv.cvtColor(input2, cv.COLOR_BGR2GRAY)
    hist2 = cv.calcHist([input2],[0],None,[256],[0,256])
    plot_img2 = cv.cvtColor(input2, cv.COLOR_BGR2RGB)
    
    
    plt.subplot(221)
    plt.imshow(plot_img)
    plt.subplot(222)
    plt.plot(hist)
    plt.subplot(223)
    plt.imshow(plot_img2)
    plt.subplot(224)
    plt.plot(hist2)
    plt.show()



def finding_pixel(input):
    mx = max(input.ravel())
    mn = min(input.ravel())
    return mx, mn


def cs(input, low, high):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
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

    new_p = new_pix.astype(np.uint8)
    img = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    new_p = cv.cvtColor(new_p, cv.COLOR_BGR2RGB)
    # img = cv.merge((input, input, input))
    # new_p = cv.merge((new_p, new_p, new_p))
    plt.subplot(121)
    plt.imshow(img)
    plt.title("original")
    plt.subplot(122)
    plt.imshow(new_p)
    plt.title("converted")
    plt.show()
    return new_p
 

#img = cv.cvtColor(cv.imread("./im/aka.jpg"), cv.COLOR_BGR2GRAY)
img = cv.imread("./im/low_contrast-man.png")
#histogram(img)
result = cs(img, 0, 255)
result2 = cs(result, 143, 253)

histogram(img, result)
histogram(result, result2)
