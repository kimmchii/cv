import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv

# path = "./im/under.png"
# path = "./im/stonk.jpg"
path = "./im/moon2.jpg"
# path = "./im/blur_circle.jpg"
# path = "./im/some_sign.jpg"

laplace = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
#if the center of laplacian coeff is negative -> use - 
#if the center of laplacian coeff is positive -> use + in this case we use + since the center is 4
laplace1 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])

x_laplace = np.array([[1,-2,1]]).reshape((1,3))
y_laplace = np.array([[1],[-2],[1]])

def inputImg(path2img):
    img = cv.resize(cv.imread(path2img), (346,398))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(cv.resize(cv.imread(path2img), (346,398)), cv.COLOR_BGR2RGB)
    return img, gray 


def sharpen(img, kernel, weight):
    filted = applyFilter(img, kernel, weight)
    # conV = np.zeros((new_img.shape[0]-kernel.shape[0]+1, new_img.shape[1]-kernel.shape[1]+1))
    conV = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i,j] + filted[i,j]
            if val <0:
                val=0
            conV[i,j] = val
    
    conV = conV.astype(np.uint8)
    return conV


def applyFilter(img, kernel, weight):
    pad_img = np.pad(img, ((1,1), (1,1)), 'constant')
    conV = np.zeros((pad_img.shape[0]- kernel.shape[0]+1, pad_img.shape[1]- kernel.shape[1]+1))
    # conV = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val =  (kernel*(pad_img[i:i+ kernel.shape[0], j:j+ kernel.shape[1]])).sum()
            conV[i,j] = val
    return conV




img, gray = inputImg(path)
filt = sharpen(gray, laplace, 1)
cv.imshow("ori", gray)
cv.imshow("filted",filt)
cv.waitKey(0)
cv.destroyAllWindows()

