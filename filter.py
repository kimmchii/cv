import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path2img = "./im/aka.jpg"
def inputImg(path2img):
    img = cv.cvtColor(cv.resize(cv.imread(path2img),(600,400)), cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img, img2

def choice():
    print("Please select: \n")
    print("1: size 3x3 kernel filter")
    print("2: size 5x5 kernel filter")
    print("3: size ?x? kernel filter (The size must be rectangle)")
    ch = int(input("Your choice is: "))
    if ch == 2:
        print("1: Gaussian filter")
        print("2: Box filter")
        print("3: Laplacian filter")
        ch = int(input("Your choice is: "))
        filter = five(ch)
        return filter
    elif ch ==1:
        print("1: Gaussian filter")
        print("2: Box filter")
        print("3: Laplacian filter") 
        ch = int(input("Your choice is: "))
        filter = three(ch)
        return filter
    elif ch==3:
        filter = convDkernel()
        return filter


def three(ch):
    match ch:
        case 1:
            Guassian = np.array([[1,2,1],[2,4,2],[1,2,1]])
            summ = Guassian.sum()
            Guassian = 1/summ*(Guassian)
            return Guassian
        case 2: 
            Box = np.array([[1,1,1], [1,1,1], [1,1,1]])
            summ = Box.sum()
            Box = 1/summ*(Box)
            return Box
        case 3:
            laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
            return laplace

def five(ch):
    match ch:
        case 1:
            Guassian =np.array([[0,1,2,1,0],[1,3,5,3,1],[2,5,9,5,2],[1,3,5,3,1],[0,1,2,1,0]])
            summ = Guassian.sum()
            Guassian = 1/summ*(Guassian)
            return Guassian
        case 2:
            Box = np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]])
            summ = Box.sum()
            Box = 1/summ*(Box)
            return Box
        case 3:
            Laplace =np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
            return Laplace

def own(kernel):
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel[i][j] = int(input("position {}x{} The value is: ".format(i+1,j+1)))
    
    summ = kernel.sum()
    if summ == 0:
        summ = 1
    kernel = 1/summ*(kernel)
    return kernel

def convDkernel():
    x,y = input("Input the size of filter: ").split()
    kernel = np.zeros((int(x), int(y)))
    kernel = own(kernel)
    print(kernel)
    return kernel

def applyFilter(img, kernel):
    conV = np.zeros((img.shape[0]- kernel.shape[0], img.shape[1]- kernel.shape[1]))
    for i in range(conV.shape[0]):
        for j in range(conV.shape[1]):
            val =  (kernel*(img[i:i+ kernel.shape[0], j:j+ kernel.shape[1]])).sum()
            if val < 0 :
                val = 0
            conV[i,j] = val
    conV = conV.astype(np.uint8)
    return conV



img, img2 = inputImg(path2img)
ch = choice()
convo = cv.cvtColor(applyFilter(img, ch ), cv.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(img2)
plt.title("Input image")
plt.subplot(122)
plt.imshow(convo)
plt.title("Choice filter")
plt.show()
# #================================Guassian==========================================

# GuassianT = np.array([[1,2,1],[2,4,2],[1,2,1]])
# summT = GuassianT.sum()
# GuassianT = 1/summT*(GuassianT)

# GuassianF =np.array([[0,1,2,1,0],[1,3,5,3,1],[2,5,9,5,2],[1,3,5,3,1],[0,1,2,1,0]])
# summF = GuassianF.sum()
# GuassianF = 1/summF*(GuassianF)
# #==================================================================================

# #================================Laplace===========================================
# LaplaceT = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
# LaplaceF =np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
# #==================================================================================

# #================================Box===============================================
# BoxT = np.array([[1,1,1], [1,1,1], [1,1,1]])
# suT = BoxT.sum()
# BoxT = 1/suT*(BoxT)

# BoxF = np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]])
# suF = BoxF.sum()
# BoxF = 1/suF*(BoxF)
# #=================================================================================

# filted1 = cv.cvtColor(applyFilter(img, GuassianF), cv.COLOR_BGR2RGB)
# filt1 = cv.cvtColor(applyFilter(img, GuassianT), cv.COLOR_BGR2RGB)

# filted2 = cv.cvtColor(applyFilter(img, BoxF), cv.COLOR_BGR2RGB)
# filt2 = cv.cvtColor(applyFilter(img, BoxT), cv.COLOR_BGR2RGB)

# filted3 = cv.cvtColor(applyFilter(img, LaplaceF), cv.COLOR_BGR2RGB)
# filt3 = cv.cvtColor(applyFilter(img, LaplaceT), cv.COLOR_BGR2RGB)

# plt.subplot(321)
# plt.imshow(img2)
# plt.title("Input image")
# plt.subplot(322)
# plt.imshow(filted1)
# plt.title("Guassian filter 5x5")
# plt.subplot(323)
# plt.imshow(img2)
# plt.title("Input image")
# plt.subplot(324)
# plt.imshow(filted2)
# plt.title("Box filter 5x5")
# plt.subplot(325)
# plt.imshow(img2)
# plt.title("Input image")
# plt.subplot(326)
# plt.imshow(filted3)
# plt.title("Laplacian filter 5x5")
# plt.show()

# plt.subplot(321)
# plt.imshow(img2)
# plt.title("Input image")
# plt.subplot(322)
# plt.imshow(filt1)
# plt.title("Guassian filter 3x3")
# plt.subplot(323)
# plt.imshow(img2)
# plt.title("Input image")
# plt.subplot(324)
# plt.imshow(filt2)
# plt.title("Box filter 3x3")
# plt.subplot(325)
# plt.imshow(img2)
# plt.title("Input image")
# plt.subplot(326)
# plt.imshow(filt3)
# plt.title("Laplacian filter 3x3")
# plt.show()