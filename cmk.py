import cv2 as cv
import numpy as np
from PIL import Image 

img = cv.resize(cv.imread("./im/aka.jpg"), (600,400))
b = img[:,:,0]/255
g = img[:,:,1]/255
r = img[:,:,2]/255  

height, width = img.shape[:2]

c = np.zeros((height,width))
m = np.zeros((height,width))
y = np.zeros((height,width))
k = np.zeros((height,width))


for i in range(height):
    for j in range(width):
        k[i][j] = 1 - max(b[i][j], g[i][j], r[i][j])
        c[i][j] = (1-r[i][j]-k[i][j])/(1-k[i][j])
        m[i][j] = (1-g[i][j]-k[i][j])/(1-k[i][j])
        y[i][j] = (1-b[i][j]-k[i][j])/(1-k[i][j])
        
k = k*255
c = c*255
m = m*255
y = y*255

CMYK_image= cv.merge((c,m,y,k)).astype(np.uint8)
# cv.imshow("CMYK Image", CMYK_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
im = Image.fromarray(CMYK_image, mode = "CMYK")
im.save('./im/test.tiff')






