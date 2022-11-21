import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv

path = "./im/i.png"

def binary_img(path2img):
    img = cv.imread(path2img,0)
    img = cv.resize(img, (int(img.shape[1]/1), int(img.shape[0]/1)))
    (thresh, im_bw) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # edges = cv.Canny(img, 100, 200)
    return im_bw

def erosion(img):
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    ero = np.array([[0,255,0], [255,255,255], [0,255,0]])
    conV = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, pad_img.shape[0]-2):
        for j in range(0, pad_img.shape[1]-2):
            if (ero[0,1]==pad_img[i,j+1])and(ero[1,0]==pad_img[i+1,j])\
                and(ero[1,1]==pad_img[i+1, j+1])and(ero[1,2]==pad_img[i+1,j+2])\
                    and(ero[2,1]==pad_img[i+2, j+1]):
                conV[i,j]=255

    return conV

def dilation(img):
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    dil = np.array([[0,255,0], [255,255,255], [0,255,0]])
    conV = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, pad_img.shape[0]-2):
        for j in range(0, pad_img.shape[1]-2):
            if (dil[0,1]==pad_img[i,j+1])or(dil[1,0]==pad_img[i+1,j])\
                or(dil[1,1]==pad_img[i+1, j+1])or(dil[1,2]==pad_img[i+1,j+2])\
                    or(dil[2,1]==pad_img[i+2, j+1]):
                conV[i,j]=255

    return conV

bw = binary_img(path)
e=erosion(bw)
d = dilation(e)
cv.imshow("e", e)
cv.imshow("ed", bw)
cv.imshow("d", d)
cv.waitKey(0)
cv.destroyAllWindows()