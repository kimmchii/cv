from tkinter.tix import MAX
import cv2 as cv
import numpy as np 



img = cv.resize(cv.imread("./im/aka.jpg"), (600,400))

b = img[:,:,0]/255
g = img[:,:,1]/255
r = img[:,:,2]/255


height, width = img.shape[:2]

mx = np.zeros((height,width))
mn = np.zeros((height,width))
h = np.zeros((height,width))
s = np.zeros((height,width))
v = np.zeros((height,width))

for i in range(height):
    for j in range(width):
        mx[i][j] = max(b[i][j], g[i][j], r[i][j])
        mn[i][j] = min(b[i][j], g[i][j], r[i][j])
        
 
        if mx[i][j] == r[i][j]:
            h[i][j] = 60*(g[i][j]-b[i][j])/(mx[i][j]-mn[i][j])
        elif mx[i][j]  == g[i][j] :
            h[i][j] = 120 + 60*(b[i][j]-r[i][j])/(mx[i][j]-mn[i][j])
        elif mx[i][j]  == b[i][j] :
            h[i][j] = 240 + 60*(r[i][j] -g[i][j] )/(mx[i][j]-mn[i][j])
        
        if mx[i][j]  == 0:
            s[i][j] = 0
        else:
            s[i][j] = (mx[i][j]-mn[i][j])/mx[i][j] 

        if h[i,j] < 0:
            h[i,j] = h[i,j] + 360

        v[i][j] = mx[i][j]


h = h/2
s = s*255
v = v*255

h = h.astype(dtype=np.uint8)
s = s.astype(dtype=np.uint8)
v = v.astype(dtype=np.uint8)
print(s)
vsh = cv.merge((h,s,v))
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
H,S,V = cv.split(hsv)

print(S)
cv.imshow("code", vsh)
cv.imshow("hsv", hsv)
cv.waitKey(0)
cv.destroyAllWindows()

