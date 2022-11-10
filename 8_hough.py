import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cv2 as cv


# path = "./im/moon2.jpg"
# path = "./im/line.png"
# path = "./im/stonk.jpg"
path = "./im/sudoku.png"



def inputImg(path2img):
    img = cv.imread(path2img)
    img = cv.resize(img, (int(img.shape[1]/1), int(img.shape[0]/1)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img, gray   

def hough(img):
    img2 = img.copy()
    blurred_image = cv.GaussianBlur(img, (9, 9), 0)
    edges = cv.Canny(blurred_image, 100,200)
    hg = cv.HoughLines(edges, 1, np.pi/180,100)
    for a in hg:
        ar = np.array(a[0], dtype=np.float64)
        rho, theta = ar
        x1 = int(np.cos(theta)*rho + 1000*(-np.sin(theta)))
        y1 = int(np.sin(theta)*rho + 1000*(np.cos(theta)))
        x2 = int(np.cos(theta)*rho - 1000*(-np.sin(theta)))
        y2 = int(np.sin(theta)*rho - 1000*(np.cos(theta)))
        cv.line(img2, (x1,y1), (x2,y2), (0,255,0), 2)
    
    cv.imshow("Original", img)
    cv.imshow("Hough", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

#     return 0 

img, gray = inputImg(path)
hough(gray)