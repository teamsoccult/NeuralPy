import matplotlib.pyplot as plt
from pylab import *
import numpy as np

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

subplots_adjust(hspace=0.000)
number_of_subplots=3

for i,v in enumerate(range(number_of_subplots)):
    for i in 1
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(x,y)

plt.show()


##needed files:
import math_helper as M
import network_helper as NH
import read_write_helper as RW
import plots_helper as P
index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}
images = RW.read_image(filename['images'])
filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = RW.read_labels(filename['images'])
###
import matplotlib.pyplot as plt
### TASK D)
import math_helper as M

def plot_images(images, labels, index_list, columns = 5): #placeholder.
    total_img = len(index_list)
    rows = math.ceil(total_img/columns)
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        cols_left = min(total_img, columns)
        if total_img < columns:
            for k in range(total_img,columns):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
            axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "binary")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].set_title(labels[index_list[(i*columns)+j]])
        total_img -= columns
    fig.tight_layout()
    plt.show()

plot_images(images, labels, index_list, 4)


index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
predictions = [7, 2, 1, 3, 4, 5, 6, 6, 8, 9, 10, 11]
filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}
images = RW.read_image(filename['images'])
filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = RW.read_labels(filename['images'])
predictions = [7, 2, 1, 3, 4, 5, 6, 6, 8, 9, 10, 11]

import math
math.ceil(2.4)

def plot_images_new(images, labels, index_list, columns = 5):
    if predictions == None:
        predictions = labels
    total_img = len(index_list)
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

plot_images_new(images, labels, index_list, predictions)
plot_images_new(images, labels, index_list)


def weights_plot(A, plt_col = 5): #weights count = integer.
    #prep.
    cols_A = M.gen_col(A)
    rows, columns = M.dim(A)
    print(f"rows: {rows}")
    print(f"cols: {columns}")

    # creating K which holds lists of 28x28.
    K = [[] for i in range(columns)]
    for i in range(columns):
        C = [[] for i in range(28)]
        for j in range(28):
            for k in range(28):
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
            axs[i,j].imshow(K[(i*plt_col)+j], cmap = "gist_heat")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].set_title((i*plt_col)+j)
        columns -= plt_col
    fig.tight_layout()
    plt.show()


network = RW.linear_load('mnist_linear.weights')
A, b = network
dim(A)


weights_plot(A,3)
