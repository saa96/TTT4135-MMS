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

def get_macro_block(img_nr, idx_y, idx_x, size=16):
    """
    Outputs a macro block with dimentions (step,step).

    Input description:
    img_nr: which image in img_data that is evaluated
    idx_y : index along y axis
    idx_x : index along x axis
    size  : size of block

    Return: Matrix with dimentions (step,step)
    """
    return img_data[img_nr][idx_y:(idx_y+size), idx_x:(idx_x+size)]

#def search_8(): # TODO: implemented for b)

def search_16(img_nr, idx_y, idx_x, size=16):
    """
    Searches in block of 32 pixels.

    Return: new coordinate of block
    """
    block_list = []
    block_list.append(get_macro_block(img_nr-1, idx_y,      idx_x,      size))  # Origin
    block_list.append(get_macro_block(img_nr-1, idx_y,      idx_x+size, size))  # Right
    block_list.append(get_macro_block(img_nr-1, idx_y+size, idx_x,      size))  # Down
    block_list.append(get_macro_block(img_nr-1, idx_y+size, idx_x+size, size))  # Right and down

    val_list = []
    current_block = get_macro_block(img_nr, idx_y, idx_x, size=16)
    for block in block_list:
        if block.shape[0] and block.shape[1]:          
            val_list.append(cost_func(current_block, block))
        else:
            val_list.append(100)
    
    displacement = np.argmin(val_list)

   #print("Block is in position %i." % displacement)

    if displacement == 0:
        return (idx_y,  idx_x)
    elif displacement == 1:
        return (idx_y,  idx_x + size)
    elif displacement == 2:
        return (idx_y + size,  idx_x)
    elif displacement == 3:
        return (idx_y + size,  idx_x + size)

    
def scan_img(img_data, img_nr,size=16):
    """
    Currently printes out the old and new position of the 16x16 block
    """
    for y in range(0,(img_data[img_nr].shape)[0],size):
        for x in range(0,(img_data[img_nr].shape)[1],size):
            new = search_16(img_nr, y, x)
            print("Old coord: (%i,%i)\nNew coord: (%i,%i)\n" % (x,y,new[1],new[0]))
    
scan_img(img_data,2)