import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv

#since we need to plot the images to the matplotlib, we need to conver the image we 
#want to plot back to rgb for true color.

#-------------Function convert BGR to grayscale----------------------------------
img = cv.resize(cv.imread("./im/aka.jpg"), (600,400))
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

gray_image_for_plot = cv.cvtColor(gray_image, cv.COLOR_GRAY2RGB)

#-------------Manual convert BGR to grayscale------------------------------------ 
#-------------luminosity method
img1 = img.copy()
gray_b = 0.114*img1[:,:,0]
gray_g = 0.587*img1[:,:,1]
gray_r = 0.299*img1[:,:,2]
gray = gray_b + gray_g + gray_r
gray = gray.astype(dtype=np.uint8)
img1[:,:,0] = gray_b + gray_g + gray_r
img1[:,:,1] = gray_b + gray_g + gray_r
img1[:,:,2] = gray_b + gray_g + gray_r
new_img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)


plt.subplot(131)
plt.title("Manual convert BGR to grayscale")
plt.imshow(gray_image_for_plot)

plt.subplot(132)
plt.title("luminosity")
plt.imshow(new_img1)
plt.show()


#-------------Function histogram----------------------------------
histr = cv.calcHist([gray_image],[0],None,[256],[0,256])


#-------------Manual histogram------------------------------------ 
def histo(image):
    hist = np.zeros(256)
    for i in image.flatten():
        hist[i] +=1
        
    return hist

x_axis = range(0,256)


#-----------Manual converted image---------------------------------
plt.subplot(231)
plt.imshow(new_img1)
plt.title("Manual converted image")
#-----------Manual histogram - Manual convert BGR to grayscale-----
hist1 = histo(gray)
plt.subplot(232)
plt.plot(x_axis, hist1)
plt.title("Manual histogram from manual method")
plt.ylabel("pixels count")
plt.xlabel("grayscale value")

#-----------Function histogram - Manual convert BGR to grayscale---
histr1 = cv.calcHist([gray],[0],None,[256],[0,256])
plt.subplot(233)
plt.plot(histr1)
plt.title("Function histogram from manual method")

plt.ylabel("pixels count")
plt.xlabel("grayscale value")


#-----------Function converted image---------------------------------
plt.subplot(234)
plt.imshow(gray_image_for_plot)
plt.title("Function converted image")
#-----------Manual histogram - function convert BGR to grayscale---
hist = histo(gray_image)
plt.subplot(235)
plt.plot(x_axis, hist)
plt.title("Manual histogram from function grayscale")
plt.ylabel("pixels count")
plt.xlabel("grayscale value")

#-----------Function histogram - function convert BGR to grayscale-
plt.subplot(236)
plt.plot(histr)
plt.title("Function histogram from function grayscale")

plt.ylabel("pixels count")
plt.xlabel("grayscale value")
plt.show()




