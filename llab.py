import cv2 as cv
import numpy as np
from PIL import Image

#we need to convert the normal image to xyz domain first


def convRGB2XYZ(path_to_photo):
    img = cv.resize(cv.imread(path_to_photo), (600,400))
    norm_b = img[:,:,0]/255
    norm_g = img[:,:,1]/255
    norm_r = img[:,:,2]/255

    height, width = img.shape[:2]
    new_r = np.zeros((height,width))
    new_g = np.zeros((height,width))
    new_b = np.zeros((height,width))
    
    for i in range(height):
        for j in range(width):
            if (norm_r[i][j] > 0.04045):
                new_r[i][j] = ((norm_r[i][j] + 0.055)/1.055)**2.4
            else:
                new_r[i][j] = norm_r[i][j]/12.92

            if (norm_g[i][j] > 0.04045):
                new_g[i][j] = ((norm_g[i][j] + 0.055)/1.055)**2.4
            else:
                new_g[i][j] = norm_g[i][j]/12.92

            if (norm_b[i][j] > 0.04045):
                new_b[i][j] = ((norm_b[i][j] + 0.055)/1.055)**2.4
            else:
                new_b[i][j] = norm_b[i][j]/12.92

    new_r = new_r*100
    new_g = new_g*100
    new_b = new_b*100

    x = new_r*0.4124 + new_g*0.3576 + new_b*0.1805
    y = new_r*0.2126 + new_g*0.7152 + new_b*0.0722
    z = new_r*0.0193 + new_g*0.1192 + new_b*0.9505
    return x, y, z, height, width


def convXYZ2LAB(x,y,z,height,width):

    x = x/95.047
    y = y/100.000
    z = z/108.883
    new_x = np.zeros((height,width))
    new_y = np.zeros((height,width))
    new_z = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            if (x[i][j] > 0.008856):
                new_x[i][j] = x[i][j]**(1/3)
            else:
                new_x[i][j] = (7.787*x[i][j]) + (16/116)

            if (y[i][j] > 0.008856):
                new_y[i][j] = y[i][j]**(1/3)
            else:
                new_y[i][j] = (7.787*y[i][j]) + (16/116)

            if (z[i][j] > 0.008856):
                new_z[i][j] = z[i][j]**(1/3)
            else:
                new_z[i][j] = (7.787*z[i][j]) + (16/116)

    cie_L = (116*new_y)-16
    cie_A = 500*(new_x-new_y)
    cie_B = 200*(new_y-new_z)

    return cie_L, cie_A, cie_B

file = "./im/aka.jpg"
x, y, z, height, weight = convRGB2XYZ(file)
L, A, B = convXYZ2LAB(x, y, z, height, weight)

LAB_image= cv.merge((L,A,B)).astype(np.uint8)
im = Image.fromarray(LAB_image, mode = "LAB")
im.save('./im/lab.tiff')