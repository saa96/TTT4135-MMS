import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

## Task 4: Prediction

# Loading and opening images
img_path = glob.glob('Program_Files/*.png')
img_data = []
for path in img_path:
    img_data.append(np.array(Image.open(path)))


## Defining functions that will go in "main"
def cost_func(curr_block,prev_block):
    return np.mean((curr_block - prev_block)**2)

def get_block(img_nr, idx_y, idx_x, size=16):
    """
    Outputs a block with dimentions (size, size).

    Input description:
    img_nr: which image in img_data that is evaluated
    idx_y : index along y axis
    idx_x : index along x axis
    size  : size of block

    Return: Matrix with dimentions (size, size)
    """
    return img_data[img_nr][idx_y:(idx_y+size), idx_x:(idx_x+size)]

def search(img_nr, idx_y, idx_x, size=16):
    """
    Searches in block of 32 pixels.

    Return: new coordinate of block
    """
    block_list = []
    coord_list = []
    v_range = 32
    # Get surrounding blocks, and their start coordinate
    for y in range(0,v_range,size):
        for x in range(0,v_range,size):
            block_list.append(get_block(img_nr-1, idx_y+y, idx_x+x, size))
            coord_list.append((idx_x+x, idx_y+y))

    # Compare the blocks aquired above with the current block
    val_list = []
    current_block = get_block(img_nr, idx_y, idx_x, size)
    for block in block_list:
        if block.shape[0] and block.shape[1]:          
            val_list.append(cost_func(current_block, block))
        else:
            val_list.append(100) # Edge "compensation"
    
    # Get index of block with lowest difference
    disp_indx = np.argmin(val_list)

    # Return the new coordinate (if the block has moved)
    return coord_list[disp_indx]

    
def scan_img(img_data, img_nr,size=16):
    """
    Returns the old coordinates of the block and its direction 
    as 4 separate matrixes.

    Input:
    img_data: list of images
    img_nr: which image in the image list that will be
            evaluated
    size: size of block (/macro block)

    Return: old_x, old_y, new_x, new_y
    """
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
            # Use search(..) on current block and get new coordinates         
            new = search(img_nr, y, x, size)
            
            # Store old coordinates of the block
            o_vec_x.append(x)
            o_vec_y.append(y)
            # Get displacement of block, pointing at previous block location
            n_vec_x.append(new[0]-x)
            n_vec_y.append(new[1]-y)

        # Add values to matrix
        orig_vec_x.append(o_vec_x)
        orig_vec_y.append(o_vec_y)
        new_vec_x.append(n_vec_x)
        new_vec_y.append(n_vec_y)
    
    # Convert matrices to numpy matrices and return them
    return np.array(orig_vec_x), np.array(orig_vec_y), np.array(new_vec_x), np.array(new_vec_y)
    

def main ():
    # Values for 16x16 macro blocks
    old_x_1, old_y_1, new_x_1, new_y_1 = scan_img(img_data,1,size=16)
    old_x_2, old_y_2, new_x_2, new_y_2 = scan_img(img_data,2,size=16)

    # Plotting of movement
    fig, ax = plt.subplots()
    ax.set_title('Movement from frame N-1 to N')
    ax.quiver(old_x_1, old_y_1, new_x_1, new_y_1,
            color="C1", angles='xy',
            scale_units='xy', scale=1, width=0.01,
            headwidth=4, minshaft=1.5)
    plt.gca().invert_yaxis()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title('Movement from frame N to N+1')
    ax.quiver(old_x_2, old_y_2, new_x_2, new_y_2,
            color="C1", angles='xy',
            scale_units='xy', scale=1, width=0.01,
            headwidth=4, minshaft=1.5)
    plt.gca().invert_yaxis()
    plt.show()

    # Values for regular 8x8 blocks
    old_x_1_8, old_y_1_8, new_x_1_8, new_y_1_8 = scan_img(img_data,1,size=8)
    old_x_2_8, old_y_2_8, new_x_2_8, new_y_2_8 = scan_img(img_data,2,size=8)    
    
    # Plotting of movement
    fig, ax = plt.subplots()
    ax.set_title('Movement from frame N-1 to N for 8x8 px blocks')
    ax.quiver(old_x_1_8, old_y_1_8, new_x_1_8, new_y_1_8,
            color="C1", angles='xy',
            scale_units='xy', scale=1, width=0.01,
            headwidth=4, minshaft=1.5)
    plt.gca().invert_yaxis()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title('Movement from frame N to N+1 for 8x8 px blocks')
    ax.quiver(old_x_2_8, old_y_2_8, new_x_2_8, new_y_2_8,
            color="C1", angles='xy',
            scale_units='xy', scale=1, width=0.01,
            headwidth=4, minshaft=1.5)
    plt.gca().invert_yaxis()
    plt.show()

main()