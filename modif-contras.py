import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv

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
    plt.title("Modif-converted image")
    plt.subplot(224)
    plt.plot(hist2)
    plt.title("Modif-converted image histogram")
    plt.show()

def finding_pixel(input):
    mx = max(input.ravel())
    mn = min(input.ravel())
    return mx, mn



def finding_quan(input):
    mx, mn = finding_pixel(input)
    pl = np.quantile(range(mn, mx+1), 0.005)
    ph = np.quantile(range(mn, mx+1), 0.995)
    return pl, ph


def modif_cs(input, low, high):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    pl, ph = finding_quan(input)
    height, width = input.shape[:2]
    new_pix = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            if (input[i][j] <= pl):
                new_pix[i][j] = low + ((input[i][j]-pl)*(high-low))/(ph-pl)

            elif(input[i][j] > pl and input[i][j] < ph):
                new_pix[i][j] = low + ((input[i][j]-pl)*(high-low))/(ph-pl)
            
            elif (input[i][j] >= pl):
                new_pix[i][j] = high + ((input[i][j]-pl)*(high-low))/(ph-pl)
    
    new_p = new_pix.astype(np.uint8)
    img = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    new_p = cv.cvtColor(new_p, cv.COLOR_BGR2RGB)

    return new_p


img = cv.imread("./im/low_contrast-man.png")
result = modif_cs(img, 0, 255)
histogram(img, result)


