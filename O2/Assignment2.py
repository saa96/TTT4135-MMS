from email.base64mime import header_length
from cv2 import mean, norm, normalize
from matplotlib import units
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
## Task 4: Prediction

img_path = glob.glob('Program_Files/*.png')
#print(img_path)
img_data = []
for path in img_path:
    img_data.append(np.array(Image.open(path)))


# Printing and testing section
empty_block = np.zeros((16,16))
macro_block_size = empty_block.shape
print("X = %i, Y = %i" %((img_data[1].shape)[1],(img_data[1].shape)[0]))
#print((img_data[1][0:8, 0:8]))#.shape)[1])
print(macro_block_size)



## Defining functions that will go in "main"
def cost_func(curr_block,prev_block):
    return np.mean(((curr_block - prev_block)**2))

def get_block(img_nr, idx_y, idx_x, size=16):
    """
    Outputs a block with dimentions (size,size).

    Input description:
    img_nr: which image in img_data that is evaluated
    idx_y : index along y axis
    idx_x : index along x axis
    size  : size of block

    Return: Matrix with dimentions (step,step)
    """
    return img_data[img_nr][idx_y:(idx_y+size), idx_x:(idx_x+size)]

def search(img_nr, idx_y, idx_x, size=16):
    """
    Searches in block of 32 pixels.

    Return: new coordinate of block
    """
    block_list = []
    coord_list = []
    v_range = int(32)
    for y in range(0,v_range,size):
        for x in range(0,v_range,size):
            block_list.append(get_block(img_nr-1, idx_y+y, idx_x+x, size))
            coord_list.append((idx_x+x,idx_y+y))

    val_list = []
    current_block = get_block(img_nr, idx_y, idx_x, size)
    for block in block_list:
        if block.shape[0] and block.shape[1]:          
            val_list.append(cost_func(current_block, block))
        else:
            val_list.append(100)
    
    disp_indx = np.argmin(val_list)

    return coord_list[disp_indx]

    
def scan_img(img_data, img_nr,size=16):
    """
    Currently printes out the old and new position of the 16x16 block
    """
    #shape_x = int(img_data[img_nr].shape[1]/size)
    #shape_y = int(img_data[img_nr].shape[0]/size)
    #print("shape x = %i, shape y = %i" % (shape_x, shape_y))
    orig_vec_x = []
    orig_vec_y = []
    new_vec_x  = []
    new_vec_y  = []
    for y in range(0,(img_data[img_nr].shape)[0],size):
        o_vec_x = []
        o_vec_y = []
        n_vec_x = []
        n_vec_y = []
        for x in range(0,(img_data[img_nr].shape)[1],size):          
            new = search(img_nr, y, x, size)

            o_vec_x.append(x)
            o_vec_y.append(y)
            n_vec_x.append(new[0]-x)
            n_vec_y.append(new[1]-y)
            #print(n_vec_x)
            #print("Old coord: (%i,%i)\nNew coord: (%i,%i)\n" % (x,y, new[0],new[1]))
        orig_vec_x.append(o_vec_x)
        orig_vec_y.append(o_vec_y)
        new_vec_x.append(n_vec_x)
        new_vec_y.append(n_vec_y)
    
    return np.array(orig_vec_x), np.array(orig_vec_y), np.array(new_vec_x), np.array(new_vec_y)
    
old_x_1, old_y_1, new_x_1, new_y_1 = scan_img(img_data,1,size=8)


fig, ax = plt.subplots()
ax.set_title('Movement from frame N-1 to N')
ax.quiver(old_x_1, old_y_1, new_x_1, new_y_1,
          color="C1", angles='xy',
          scale_units='xy', scale=1, width=0.01,
          headwidth=4, minshaft=1.5)
plt.gca().invert_yaxis()
plt.show()