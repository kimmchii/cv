import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv

# path = "./im/i.png"
# path = "./im/i_dot_inside.png"
# path2 = "./im/i_dot_outside.png"
path = "./im/incred_3.png"
def binary_img(path2img):
    img = cv.imread(path2img,0)
    img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    (thresh, im_bw) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    im = cv.cvtColor(im_bw, cv.COLOR_BGR2RGB)
    return im_bw, im

def erosion(img):
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    ero = np.array([[255,255,255], [255,255,255], [255,255,255]])
    conV = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, pad_img.shape[0]-2):
        for j in range(0, pad_img.shape[1]-2):
            if (ero[0,1]==pad_img[i,j+1])and(ero[1,0]==pad_img[i+1,j])\
                and(ero[1,1]==pad_img[i+1, j+1])and(ero[1,2]==pad_img[i+1,j+2])\
                    and(ero[2,1]==pad_img[i+2, j+1])and(ero[0,0]==pad_img[i,j])and(ero[0,2]==pad_img[i, j+2])\
                        and(ero[2,0]==pad_img[i+2, j])and(ero[2,2]==pad_img[i+2, j+2]):
                conV[i,j]=255
    conV = conV.astype(np.uint8)
    return conV

def dilation(img):
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    dil = np.array([[255,255,255], [255,255,255], [255,255,255]])
    conV = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, pad_img.shape[0]-2):
        for j in range(0, pad_img.shape[1]-2):
            if (dil[0,1]==pad_img[i,j+1])or(dil[1,0]==pad_img[i+1,j])\
                or(dil[1,1]==pad_img[i+1, j+1])or(dil[1,2]==pad_img[i+1,j+2])\
                    or(dil[2,1]==pad_img[i+2, j+1])or(dil[0,0]==pad_img[i,j])or(dil[0,2]==pad_img[i,j+2])\
                        or(dil[2,0]==pad_img[i+2,j])or(dil[2,2]==pad_img[i+2,j+2]):
                conV[i,j]=255
    conV = conV.astype(np.uint8)
    return conV

def opening(path):
    kernel = np.ones((5,5),np.uint8)
    bw, im = binary_img(path)
    erode = erosion(bw)
    open = dilation(erode)
    open_CV = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    plt.subplot(131)
    plt.imshow(cv.cvtColor(bw, cv.COLOR_BGR2RGB))
    plt.title("Original bianry image")
    plt.subplot(132)
    plt.imshow(cv.cvtColor(open_CV, cv.COLOR_BGR2RGB))
    plt.title("OpenCV function")
    plt.subplot(133)
    plt.imshow(cv.cvtColor(open, cv.COLOR_BGR2RGB))
    plt.title("Opening developed function")
    plt.show()
    return open, im

def closing(path):
    kernel = np.ones((5,5),np.uint8)
    bw, im = binary_img(path)
    dilate = dilation(bw)
    close = erosion(dilate)
    open_CV = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)
    plt.subplot(131)
    plt.imshow(cv.cvtColor(bw, cv.COLOR_BGR2RGB))
    plt.title("Original bianry image")
    plt.subplot(132)
    plt.imshow(cv.cvtColor(open_CV, cv.COLOR_BGR2RGB))
    plt.title("OpenCV function")
    plt.subplot(133)
    plt.imshow(cv.cvtColor(close, cv.COLOR_BGR2RGB))
    plt.title("Closing developed function")
    plt.show()
    return open, im

closing(path)
opening(path)

