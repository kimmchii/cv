import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

H0 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
H1 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
H2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
H3 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
D0 = [H0, H1 , H2, H3]

H4 = -1*H0
H5 = -1*H1
H6 = -1*H2
H7 = -1*H3
D1 = [H4, H5, H6, H7]

# path = "./im/under.png"
# path = "./im/stonk.jpg"
path = "./im/mount.jpg"
def inputImg(path2img):
    img = cv.resize(cv.imread(path2img), (500,500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(cv.resize(cv.imread(path2img), (500,500)), cv.COLOR_BGR2RGB)
    return img, gray 

def applyFilter(img, kernel1, kernel2):
    conV = np.zeros((img.shape[0]- kernel1.shape[0]+1, img.shape[1]- kernel1.shape[1]+1))
    conV1 = np.zeros((img.shape[0]- kernel1.shape[0]+1, img.shape[1]- kernel1.shape[1]+1))
    conV2 = np.zeros((img.shape[0]- kernel2.shape[0]+1, img.shape[1]- kernel2.shape[1]+1))
    
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
            
            if val < 0:
                val = 0
            else:
                val = val
            conV[i,j] = val
            conV = conV.astype(np.uint8)
    return conV

def compassFilter(img, array1, array2):
    if(len(array1)==len(array2)):
        conV=[None]*len(array1)
        for i in range(len(array1)):
            conV[i] = applyFilter(img, array1[i], array2[i])

        filt_img = np.zeros((conV[0].shape[0], conV[0].shape[1]))
        for i in range(conV[0].shape[0]):
            for j in range(conV[0].shape[1]):
                filt_img[i,j] = max(conV[0][i,j], conV[1][i,j], conV[2][i,j], conV[3][i,j])

    filt_img = filt_img.astype(np.uint8)
    return filt_img


img, gray = inputImg(path)
fit = compassFilter(gray, D0, D1)
fit = cv.cvtColor(fit, cv.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(122)
plt.imshow(fit)
plt.title("Applied Compass Filter")
plt.show()






