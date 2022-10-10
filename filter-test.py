import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def show_image(bgr_image):
    converted_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    plt.imshow(converted_image)

img_path = Path("./im/lucy.png").__str__()

img = cv2.imread(img_path)
img_grayed = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (400,400))

def perform_convolution(input_img, H, weight_choice):

    img_size = input_img.shape
    img_copy = np.copy(input_img)

    X = int(H.shape[0] / 2); Y = int(H.shape[1] / 2)
    print
    print(f"X: {X}"); print(f"Y: {Y}")
    print(f"Centerpoint: {H[X][Y]}")

    if ((H.flatten().sum()) != 0):
        print(f"Weight: {H.flatten().sum()}")
        ext_filter_coeff = 1 / H.flatten().sum()
    
    else:
        ext_filter_coeff = 1

    print(f"Filter Co-eff: {ext_filter_coeff}")

    print(H)
    output_width = (int(img_size[0] - X + 0))
    output_height = (int(img_size[1] - Y + 0))

    res = np.zeros(((output_width), (output_height)))
    
    for i in range(0, output_width):
        for j in range(0, output_height ):
            convolution_sum = 0
            for k in range(0, H.shape[0]):
                for l in range(0, H.shape[1]):
                    pixel_values = img_copy[i + k][j + l]
                    filter_coeff = H[l][k]
                    convolution_sum = convolution_sum + filter_coeff * pixel_values

    #         prelim_res = (convolution_sum * ext_filter_coeff)

    #         if (prelim_res < 0):
    #             prelim_res = 0
            
    #         res[i][j] = prelim_res
    # print(res.shape)
    return res.astype(np.uint8)

H = np.array([
    [1, 2, 1], 
    [2, 40, 2], 
    [1, 2, 1]
    ])

Maxican =np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
Guassian =np.array([[0,1,2,1,0],[1,3,5,3,1],[2,5,9,5,2],[1,3,5,3,1],[0,1,2,1,0]])
# H = np.array([
#     [0, 0, 0], 
#     [0, 1, 0], 
#     [0, 0, 0]
#     ])

res = perform_convolution(img_grayed, Guassian, 'idk')
print(f"\nORIGINAL: \n{img_grayed}\n")
print(f"\nRES: \n {res}\n")


plt.subplot(121)
show_image(img_grayed)

plt.subplot(122)
show_image(res)
plt.show()