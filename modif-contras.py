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
    plt.subplot(121)
    plt.imshow(img)
    plt.title("original")
    plt.subplot(122)
    plt.imshow(new_p)
    plt.title("converted")
    plt.show()
    return new_p

def finding_quan(input):
    mx, mn = finding_pixel(input)
    print(mn, mx)
    pl = np.quantile(range(mn, mx+1), 0.005)
    ph = np.quantile(range(mn, mx+1), 0.995)
    print(ph, pl)
    return pl, ph


def modif_conf(input, low, high):
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
    plt.subplot(121)
    plt.imshow(img)
    plt.title("original")
    plt.subplot(122)
    plt.imshow(new_p)
    plt.title("converted")
    plt.show()
    return new_p


#img = cv.resize(cv.imread("./im/aka.jpg"), (600,400))
img = cv.imread("./im/low_contrast-man.png")
result = modif_conf(img, 0, 255)
#result2 = modif_conf(result, 0, 255)
histogram(img, result)


