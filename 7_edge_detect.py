import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path = "./im/lucy.png"
# path = "./im/maki.jpeg"
def inputImg(path2img):
    img = cv.resize(cv.imread(path), (500,500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray 

maskX = 1/6*np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
mask_X_sobel = 1/6*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
mask_Y_sobel = 1/6*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
mask_X_prewitt = 1/8*np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
mask_Y_prewitt = 1/8*np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
# maskX_b = np.array([-1,0,1])
# maskX_a = np.array([[1],[1],[1]])
# print(maskX_a*maskX_b)

maskY = 1/8*np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
# maskY_a = np.array([1,1,1])
# maskY_b = np.array([[-1],[0],[1]])
# print(maskY_a*maskY_b)

# def test(img, kernel):
#     conV = np.zeros((img.shape[0]-kernel.shape[0], img.shape[1]))
#     for i in range(conV.shape[1]):
#         for j in range(conV.shape[0]):
#             val =  (1/2*kernel*(img[j:j+kernel.shape[0], i])).sum()
#             # print(val)
#             if val < 5 :
#                 conV[j,i] = 0
#             else :
#                 conV[j,i] = 255
#     conV = conV.astype(np.uint8)
#     return conV

def applyFilter(img, kernel1, kernel2, threshold):
    conV = np.zeros((img.shape[0]- kernel1.shape[0], img.shape[1]- kernel1.shape[1]))
    conV1 = np.zeros((img.shape[0]- kernel1.shape[0], img.shape[1]- kernel1.shape[1]))
    conV2 = np.zeros((img.shape[0]- kernel2.shape[0], img.shape[1]- kernel2.shape[1]))
    
    for i in range(conV1.shape[0]):
        for j in range(conV1.shape[1]):
            val =  (kernel1*(img[i:i+ kernel1.shape[0], j:j+ kernel1.shape[1]])).sum()
            if val < 0 :
                val = 0
            conV1[i,j] = val
    for i in range(conV2.shape[0]):
        for j in range(conV2.shape[1]):
            val =  (kernel2*(img[i:i+ kernel2.shape[0], j:j+ kernel2.shape[1]])).sum()
            if val < 0 :
                val = 0
            conV2[i,j] = val
    conV1 = conV1.astype(np.uint8)
    conV2 = conV2.astype(np.uint8)

    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val = np.sqrt(conV1[i,j]**2 + conV2[i,j]**2)
            
            if val < threshold:
                val = 0
            else:
                val = 255
            conV[i,j] = val
            conV = conV.astype(np.uint8)
    return conV

img, gray = inputImg(path)
conV= applyFilter(gray, mask_X_sobel, mask_Y_sobel, 10)
conV2 = applyFilter(gray, mask_X_prewitt, mask_Y_prewitt, 10)
cv.imshow("dd", conV)
cv.imshow("ff", conV2)
cv.waitKey(0)
cv.destroyAllWindows()
