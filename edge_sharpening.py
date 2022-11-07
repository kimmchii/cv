import numpy as np
import cv2
import matplotlib.pyplot as plt

# Edge sharpening
def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.resize(cv2.imread("./im/moon.jpg",0), (1000,1000))

laplacian = np.multiply(np.array([[0,1,0],[1,-4,1],[0,1,0]]), (1/8))

def sizing(image,filters):
    out_size = (image.shape[0] - filters.shape[0]) + 1
    return out_size

def apply_laplace(image,kernel):
    outputsize = sizing(image,kernel)
    filtered = np.zeros((outputsize,outputsize))
    for i in range (outputsize):
        for j in range (outputsize):
            pre_res = (kernel * image[i:i+kernel.shape[0], j:j+kernel.shape[1]]).sum()
            # if (pre_res < 0)
            # pre_res = 0
            filtered[i][j] = pre_res
    return filtered.astype(np.uint8)

def edge_sharpening(input,kernel_filter):
    results = np.zeros((input.shape[0],input.shape[1]))
    derivate = apply_laplace(input,kernel_filter)
    derivate_pad = np.pad(derivate, (2,2), 'edge')
    for u in range (input.shape[0]):
        for v in range (input.shape[1]):
            pre_out = input[u][v] - derivate_pad[u][v]
            if (pre_out < 0):
                pre_out = 0
            results[u][v] = pre_out
    return results.astype(np.uint8)

edge_sharpen = edge_sharpening(img,laplacian)


hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([edge_sharpen], [0], None, [256], [0, 256])

# print(img.shape[0],img.shape[1])
# print(edge_sharpen.shape[0],edge_sharpen.shape[1])
plt.figure(1)
plt.subplot(1,2,1)
plt.title("Original image")
convert_color(img)
plt.subplot(1,2,2)
plt.title("Edge sharpen")
convert_color(edge_sharpen)

plt.figure(2)
plt.subplot(1,2,1)
plt.title("Original image: histogram")
plt.plot(hist1)
plt.subplot(1,2,2)
plt.title("Edge sharpen: histogram")
plt.plot(hist2)

plt.show()


