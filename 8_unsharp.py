import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

path = "./im/moon2.jpg"

laplace = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
#if the center of laplacian coeff is negative -> use - 
#if the center of laplacian coeff is positive -> use + in this case we use + since the center is 4
laplace1 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])

guassian = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])



def inputImg(path2img):
    img = cv.resize(cv.imread(path2img), (346,398))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(cv.resize(cv.imread(path2img), (346,398)), cv.COLOR_BGR2RGB)
    return img, gray 


def blur(img, kernel):
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    conV = np.zeros((pad_img.shape[0]- kernel.shape[0]+1, pad_img.shape[1]- kernel.shape[1]+1))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val =  (kernel*(pad_img[i:i+ kernel.shape[0], j:j+ kernel.shape[1]])).sum()
            if val<0:
                val = 0
            conV[i,j] = val
    conV = conV.astype(np.uint8)
    return conV

def unsharpen(img, smooth, weight):
    smoothed = blur(img, smooth)
    mask = img-smoothed
    new_img = img+ (weight*mask)
    new_img = new_img.astype(np.uint8)
    return new_img



img, gray = inputImg(path)
conV = unsharpen(gray, guassian, 3)
hist1 = cv.calcHist([gray], [0], None, [256], [0, 256])
hist2 = cv.calcHist([conV], [0], None, [256], [0, 256])

cv.imshow("d", gray)
cv.imshow("f", conV)
cv.waitKey(0)
cv.destroyAllWindows()


plt.show()