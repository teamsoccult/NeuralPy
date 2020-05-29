'''
TASKS: D, N, O
'''
import matplotlib.pyplot as plt
import math_helper as M
import math
### TASK D)

def plot_images(images, labels, index_list = 10, columns = 5):

    '''
    Description:
    Returns multiple sub-plots of images.

    Assumes index_list has a length which is divisible by 5
    (e.g., has length 5, 10, 15, 20, etc.).

    Using the subplots function from matplotlib.pyplot we always
    plot 5 columns, and add additional rows depending on the
    size of the input. The function uses imshow to display the
    pixel values as an image. The colormap "binary" is used, 
    which is a black and white representation and makes for easy
    deciphering of the digits. The title will be the label
    associated with the particular image. For aesthetic purposes
    we have removed the axis ticks and values, as these don't
    actually correspond to anything meaningful in this case.

    ________

    Arguments:
    images = list with any number of pixel images (usually 28x28).
    labels = list with any number of labels (e.g., '7') corresponding to images.
    index_list = list containing indexes of which images/labels to plot.

    '''
    #first, test whether index list has been specified.
    if isinstance(index_list, list):
        total_img = len(index_list)
    else:
        total_img = index_list

    rows = math.ceil(total_img/columns)
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        cols_left = min(total_img, columns)
        if total_img < columns:
            for k in range(total_img,columns):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
            if isinstance(index_list, list):
                axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "binary")
                axs[i,j].set_title(labels[index_list[(i*columns)+j]])
            else:
                axs[i,j].imshow(images[(i*columns)+j], cmap = "binary")
                axs[i,j].set_title(labels[(i*columns)+j])
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
        total_img -= columns
    fig.tight_layout()
    plt.show()

### TASK N)

def plot_images_new(images, labels, index_list = 10, columns = 5, predictions = None):
    if isinstance(index_list, list):
        total_img = len(index_list)
    else:
        total_img = index_list

    if predictions == None:
        predictions = labels
        
    rows = math.ceil(total_img/columns)
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        cols_left = min(total_img, columns)
        if total_img < columns:
            for k in range(total_img, columns):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
            axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "binary")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            if labels[i+j] == predictions[i+j]:
                axs[i,j].set_title(predictions[index_list[(i*columns)+j]])
            else:
                axs[i,j].imshow(images[index_list[i+j]], cmap = "Reds")
                axs[i,j].set_title(f'{predictions[index_list[(i*columns)+j]]}, correct {labels[index_list[i+j]]}', color = 'red')
        total_img -= columns
    fig.tight_layout()
    plt.show()

### TASK O)

def weights_plot(A, plt_col = 5, image_dim = 28): #weights count = integer.
    #prep.
    cols_A = M.gen_col(A)
    rows, columns = M.dim(A)

    # creating K which holds lists of 28x28.
    K = [[] for i in range(columns)]
    for i in range(columns):
        C = [[] for i in range(image_dim)]
        for j in range(image_dim):
            for k in range(image_dim):
                C[j].append(next(cols_A))
        K[i].append(C)

    K = [y for x in K for y in x] #flatten the list.
    #needed for the plot:
    plt_row = math.ceil(columns/plt_col)
    fig, axs = plt.subplots(plt_row, plt_col)

    #plotting
    for i in range(plt_row):
        cols_left = min(columns, plt_col)
        if columns < plt_col:
            for k in range(columns, plt_col):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
            axs[i,j].imshow (K[(i*plt_col)+j], cmap = "gist_heat")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].set_title((i*plt_col)+j)
        columns -= plt_col
    fig.tight_layout()
    plt.show()
