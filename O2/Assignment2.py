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


# Printing and testing section
empty_block = np.zeros((16,16))
macro_block_size = empty_block.shape
print("X = %i, Y = %i" %((img_data[1].shape)[1],(img_data[1].shape)[0]))
print((img_data[1][0:8, 0:8]))#.shape)[1])
print(macro_block_size)



## Defining functions that will go in "main"
def cost_func(curr_block,prev_block):
    return np.mean(curr_block - prev_block)**2

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
    for y in range(0,32,size):
        for x in range(0,32,size):
            block_list.append(get_block(img_nr-1, idx_y+y, idx_x+x, size))
            coord_list.append((idx_x+x,idx_y+y))

    val_list = []
    current_block = get_block(img_nr, idx_y, idx_x, size=16)
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
    orig_vec = []
    new_vec  = []
    for y in range(0,(img_data[img_nr].shape)[0],size):
        o_vec = []
        n_vec = []
        for x in range(0,(img_data[img_nr].shape)[1],size):
            
            new = search(img_nr, y, x)
            o_vec.append((x,y))
            n_vec.append(new)
            #print("Old coord: (%i,%i)\nNew coord: (%i,%i)\n" % (x,y, new[0],new[1]))
        orig_vec.append(o_vec)
        new_vec.append(n_vec)
    
    return orig_vec, new_vec
    
old, new = scan_img(img_data,1)

# plt.quiver(old, new) # TODO: Figure out how to use this
# plt.show()