import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

path2img = "./im/pepper_girl.png"
mask = np.zeros(shape=(3,3))
W_mask = np.array([[1,2,1], [2,3,2], [1,2,1]])

def inputImg(path2img):
    #img = cv.cvtColor(cv.resize(cv.imread(path2img),(600,400)), cv.COLOR_BGR2GRAY)
    img0 = cv.imread(path2img, 0)
    img1 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
    return img0, img1

img0, img1 = inputImg(path2img)
# cv.imshow("", img0)
# cv.waitKey(0)
# cv.destroyAllWindows()
def MedianF_Apply(img):
    conV = np.zeros((img.shape[0]- mask.shape[0], img.shape[1]- mask.shape[1]))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val =  np.median(img[i:i+ mask.shape[0], j:j+ mask.shape[1]])
            conV[i,j] = val
    conV = conV.astype(np.uint8)
    return conV

def MaximumF_Apply(img):
    conV = np.zeros((img.shape[0]- mask.shape[0], img.shape[1]- mask.shape[1]))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val =  np.amax(img[i:i+ mask.shape[0], j:j+ mask.shape[1]])
            conV[i,j] = val
    conV = conV.astype(np.uint8)
    return conV

def MinimumF_Apply(img):
    conV = np.zeros((img.shape[0]- mask.shape[0], img.shape[1]- mask.shape[1]))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val =  np.amin(img[i:i+ mask.shape[0], j:j+ mask.shape[1]])
            conV[i,j] = val
    conV = conV.astype(np.uint8)
    return conV

def W_MedianF_Apply(img, W):
    conV = np.zeros((img.shape[0]- W.shape[0], img.shape[1]- W.shape[1]))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            vec = []
            n_img = img[i:i+W.shape[0], j:j+W.shape[1]]
            for m in range(n_img.shape[0]):
                for n in range(n_img.shape[1]):
                    for x in range(W[m][n]):
                    #print(n_img[m][n]*W[m][n])
                        vec.append(n_img[m][n])
            conV[i,j] = np.median(np.array(sorted(vec)))
    conV = conV.astype(np.uint8)
    return conV


img0, img1 = inputImg(path2img)
median = cv.cvtColor(MedianF_Apply(img0), cv.COLOR_BGR2RGB)
maximum = cv.cvtColor(MaximumF_Apply(img0), cv.COLOR_BGR2RGB)
minimum = cv.cvtColor(MinimumF_Apply(img0), cv.COLOR_BGR2RGB)
w_median= cv.cvtColor(W_MedianF_Apply(img0, W_mask), cv.COLOR_BGR2RGB)

# Median filter
plt.subplot(121)
plt.imshow(img1)
plt.title("Original image")
plt.subplot(122)
plt.imshow(median)
plt.title("Applying Median filter")
plt.show()

#Maximum filter
plt.subplot(121)
plt.imshow(img1)
plt.title("Original image")
plt.subplot(122)
plt.imshow(maximum)
plt.title("Applying Maximum filter")
plt.show()

#Minimum filter
plt.subplot(121)
plt.imshow(img1)
plt.title("Original image")
plt.subplot(122)
plt.imshow(minimum)
plt.title("Applying Minimum filter")
plt.show()

#Weighted filter
plt.subplot(121)
plt.imshow(img1)
plt.title("Original image")
plt.subplot(122)
plt.imshow(w_median)
plt.title("Applying Weighted Median filter")
plt.show()

