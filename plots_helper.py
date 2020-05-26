'''
TASKS: D, N, O
'''
import matplotlib.pyplot as plt
import math_helper as M
### TASK D)

def plot_images(images, labels, index_list):

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

    rows = len(index_list) // 5
    columns = 5
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        for j in range(columns):
            axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "binary")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].set_title(labels[index_list[(i*columns)+j]])
    plt.show()


### TASK N)

def plot_images_new(images, labels, index_list, predictions):
    rows = len(index_list) // 5
    columns = 5
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        for j in range(columns):
            axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "binary")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            if labels[i+j] == predictions[i+j]:
                axs[i,j].set_title(predictions[index_list[(i*columns)+j]])
            else:
                axs[i,j].imshow(images[index_list[i+j]], cmap = "Reds")
                axs[i,j].set_title(f'{predictions[index_list[(i*columns)+j]]}, correct {labels[index_list[i+j]]}', color = 'red')
    fig.tight_layout(pad=2.0)
    plt.show()

### TASK O)

def weights_plot(A):
    #prep.
    cols_A = M.gen_col(A)
    rows, columns = M.dim(A)

    # creating K which holds lists of 28x28.
    K = [[] for i in range(10)]
    for i in range(columns):
        C = [[] for i in range(28)]
        for j in range(28):
            for k in range(28):
                C[j].append(next(cols_A))
        K[i].append(C)

    K = [y for x in K for y in x] #flatten the list.

    #needed for the plot:
    col_plt = 5
    row_plt = 2
    fig, axs = plt.subplots(2, 5)

    #plotting
    for i in range(row_plt):
        for j in range(col_plt):
            axs[i,j].imshow(K[(i*col_plt)+j], cmap = "gist_heat")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].set_title((i*col_plt)+j)
    plt.show()