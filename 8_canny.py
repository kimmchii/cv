import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage
#Canny edge detection steps
# path = "./im/moon2.jpg"
path = "./im/mount.jpg"

def inputImg(path2img):
    img = cv.imread(path2img)
    img = cv.resize(img, (int(img.shape[1]/6), int(img.shape[0]/6)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img, gray

#1. Smooth the image
def smooth(img):
    guassian = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    conV = np.zeros((pad_img.shape[0]- guassian.shape[0]+1, pad_img.shape[1]- guassian.shape[1]+1))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val =  (guassian*(pad_img[i:i+ guassian.shape[0], j:j+ guassian.shape[1]])).sum()
            if val<0:
                val = 0
            conV[i,j] = val
    conV = conV.astype(np.uint8)
    return conV


#2. Gradient calculation
#In this step, we may find the orientation and edge strength
def GradientC(img):
    sobel_x = 1/16*np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = 1/16*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    gd = np.zeros((pad_img.shape[0]- sobel_x.shape[0]+1, pad_img.shape[1]- sobel_x.shape[1]+1))
    theta = np.zeros((pad_img.shape[0]- sobel_x.shape[0]+1, pad_img.shape[1]- sobel_x.shape[1]+1))
    conVx = np.zeros((pad_img.shape[0]- sobel_x.shape[0]+1, pad_img.shape[1]- sobel_x.shape[1]+1))
    conVy = np.zeros((pad_img.shape[0]- sobel_y.shape[0]+1, pad_img.shape[1]- sobel_y.shape[1]+1))

    for i in range(conVx.shape[0]):
        for j in range(conVx.shape[1]):
            val =  (sobel_x*(pad_img[i:i+ sobel_x.shape[0], j:j+ sobel_x.shape[1]])).sum()
            if val < 0 :
                val = 0
            conVx[i,j] = val
    for i in range(conVy.shape[0]):
        for j in range(conVy.shape[1]):
            val =  (sobel_y*(pad_img[i:i+ sobel_y.shape[0], j:j+ sobel_y.shape[1]])).sum()
            if val < 0 :
                val = 0
            conVy[i,j] = val

    conVx = conVy.astype(np.uint8)
    conVy = conVy.astype(np.uint8)
    # gd = np.sqrt(conVx*conVx + conVy*conVy)
    for i in range(gd.shape[0]):
        for j in range(gd.shape[1]):
            val = np.sqrt(conVx[i,j]**2+conVy[i,j])
            gd[i,j] = val

    # gd = np.hypot(conVx, conVy)
    for i in range(conVx.shape[0]):
        for j in range (conVx.shape[1]):
            val = np.arctan2(conVx[i,j], conVy[i,j])
            theta[i,j] = val
    # theta = np.arctan2(conVx, conVy)
    gd = gd.astype(np.uint8)
 
    return gd, theta

#3. Non-maximum suppression
# In this step, we amy check all the 8 pixel around the interested one
def suppress(img, theta):
    converted = np.zeros((img.shape[0], img.shape[1]))
    theta = (theta*180)/np.pi

    for i in range(1, converted.shape[0]-1):
        for j in range(1, converted.shape[1]-1):
            if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] < 180):
                a = img[i, j+1]
                b = img[i, j-1]
            elif (22.5 <= theta[i,j] < 67.5):
                a = img[i+1, j-1]
                b = img[i-1, j+1]
            elif (67.5 <= theta[i,j] < 112.5):
                a = img[i+1, j]
                b = img[i-1, j]
            elif (112.5 <= theta[i,j] < 157.5):
                a = img[i-1, j-1]
                b = img[i+1, j+1]
            
            if(img[i,j] >= a) and (img[i,j] >= b):
                converted[i,j] = img[i,j]
            else:
                converted[i,j] = 0
     
    converted = converted.astype(np.uint8)
    return converted
        
#4. Double threshold
def doubleT(img, ht, lt):
    highest = img.max()*ht
    lowest = highest*lt
    result = np.zeros((img.shape[0], img.shape[1]))

    strong_x, strong_y = np.where(img >= highest)
    # zero_x, zero_y = np.where(img < lowest)
    weak_x, weak_y = np.where((lowest <= img)&(img<=highest))

    result[strong_x, strong_y] = 255
    result[weak_x, weak_y] = 25
    # result[zero_x, zero_y] = 0
    
    result = result.astype(np.uint8)
    return result

#5. Edge tracking by Hysteresis
def hyst(img):
    res = np.zeros((img.shape[0], img.shape[1]))
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if(img[i,j]==25):
                if((img[i+1, j-1] == 255) or (img[i+1, j] == 255) or (img[i+1, j+1] == 255)
                        or (img[i, j-1] == 255) or (img[i, j+1] == 255)
                        or (img[i-1, j-1] == 255) or (img[i-1, j] == 255) or (img[i-1, j+1] == 255)):
                    res[i,j] = 255
                else: 
                    res[i,j] = 0

            else:
                res[i,j] = img[i,j]
    res = res.astype(np.uint8)
    return res

# conclue
def cann(path):
    img, gray = inputImg(path)
    conV = smooth(gray)
    gd,theta = GradientC(conV)
    converted = suppress(gd, theta)
    doble = doubleT(converted, 0.05, 0.09)
    res = hyst(doble)
    return res

img = cv.imread(path)
img = cv.resize(img, (int(img.shape[1]/6), int(img.shape[0]/6)))
edges = cv.Canny(img,100,200)
res = cann(path)

cv.imshow("c", res)
cv.imshow("orginal", edges)
cv.waitKey(0)
cv.destroyAllWindows()