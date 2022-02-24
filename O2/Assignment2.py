import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
## Task 4: Prediction

img_path = glob.glob('Program_Files/*.png')
#print(img_path)
img_data = []
for path in img_path:
    img_data.append(np.array(Image.open(path)))


## Image data testing:
# print(img_data[0])
# plt.figure()
# plt.imshow(img_data[0])
search_range = (32,32)
empty_block = np.zeros((16,16))
block_size = empty_block.shape
print(block_size)

# This wont work, its just a "sketch" implementation
def cost_func(curr_img,prev_img,X,v):
    return np.mean(curr_img(X) - prev_img(X-v))

def opt_disp(img_data,block,serch_range):
    disp_array = np.zeros_like(img_data[0])
    for i in range(img_data[0].shape[0]):
        for j in range(img_data[0].shape[1]):
            disp_array[i][j] = cost_func(img_data[1],img_data[0],block,serch_range)

    disp = np.min(np.sum(disp_array))
    return disp
