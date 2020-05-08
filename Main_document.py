import struct as st
import matplotlib.pyplot as plt

### TASK B)
### TSK B)

filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}

def read_labels(filename):
    '''Using the struct.unpack function, this function reads the labels from
    MNIST-data. Assumes that the file is in the same folder as this python document.
    Assumes that the number of labels is 10000.
    Returns list of the labels, and a print message if the magic number is 2049.
    '''
    with open(filename, 'rb') as f:
        f.seek(2) #start after the 2 zeroes
        magic, zeros2, no_items = st.unpack('>HHH',f.read(6)) #magic number as hex digit
        if magic == 2049:
            print(f'The Magic Number is {magic}!')
        else:
            print(f"The magic number is {magic}, which is not 2049.")
        f.seek(8)
        labels = st.unpack(f">{no_items}B", f.read(no_items))
        return list(labels)

labels = read_labels(filename['images'])
labels

### TASK C)

filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}

def read_image(filename):
    '''Using the struct.unpack function, this function reads the image data
    from the MNIST-database. Assumes that the data is stored in the same folder
    as this document.
    Returns a list of the image data and print message if the magic number is 2051.
    '''
    with open(filename, 'rb') as f:
        f.seek(2) #start after the 2 zeroes
        magic, zeros2, noIm, zeros3, noR, zeros4, noC = st.unpack('>HHHHHHH', f.read(14)) #magic number as hex digit
        if magic == 2051:
            print(f'The Magic Number is {magic}!')
        else:
            print(f"The Magic Number is {magic}, and not 2051")

        images = list()

        for i in range(noIm):
            image = list()
            for j in range(noR):
                row = list(st.unpack(">28B", f.read(28)))
                image.append(row)
            images.append(image)

    return images

images = read_image(filename['images'])

### IMPORT MATPLOTLIB.PYPLOT AS PLT:
labels[0]

### D): (variables we need)
index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}
images = read_image(filename['images'])
filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = read_labels(filename['images'])

#assumes input is divisor of 5.
def plot_images_new(images, labels, index_list):
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

plot_images_new(images, labels, index_list)

### F):
import json
import os

filename = "mnist_linear.weights"

def linear_load(filename):
    try:
        with open(filename, "r") as f:
            json_string = json.load(f)
            return json_string
    except FileNotFoundError:
        print(f"Cannot find {filename} in the directory. \nPlease check the filename and the pathing to said filename.")

json_string = linear_load(filename)

### RETURN TO LINEAR SAVE - CONSIDER ALSO MKAING FILENAME A STRING IF IT IS NOT:

def linear_save(filename, network): ## inspired by https://stackoverflow.com/questions/42718922/how-to-stop-overwriting-a-file?fbclid=IwAR3osjuyuJTJtvP9wqpBsuBQz8WWTlKmOSpgAmMhn5qXETZ6po7m58GHyAA
    flag = True
    while flag:
        if os.path.isfile(filename):
            response = input(f"There is already a file named {filename}\nOverwrite? (Yes/No)")
            if response.lower() != "yes":
                break
        network = json.dumps(network)
        with open(filename, "w") as f:
            f.write(network)
            flag = False

linear_save('tommy', json_string)

### G):
def image_to_vector(image): #inspired by https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    '''Standardize image to be a single list (image vector).
    Assumes that the values of the image is between [0, 255].
    Returns a list of floats between [0,1]. '''
    return [(item)/(255) for sublist in image for item in sublist]

### H)

### IMPORTING MATRIX CLASS FROM MATRIX CLASS DOCUMENT - CALLING IT M:

## https://stackoverflow.com/questions/17531796/find-the-dimensions-of-a-multidimensional-python-array ##

import matrix_functions2 as M

#also assert that both should be of equal length
def mean_square_error(U, V):
    if not isinstance(U, list) or not isinstance(V, list):
        raise TypeError("Input must be lists.")
    vector_sum = 0
    for i in range(len(U)):
        vector_sum += (V[i]-U[i])**2
    return vector_sum/len(U)

### CHECK EXAMPLE:
mean_square_error([1,2,3,4], [3,1,3,2]) #checks out

### CHECK ASSERTIONS
#mean_square_error([1,2,3,4], 5) #checks out

### J):
V = [1,2,3,4]

def argmax(V): ### inspired by https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
    if not isinstance(V, list):
        raise TypeError("Input must be a list.")

    return V.index(max(V))

#CHECK EXAMPLE:
argmax([6, 2, 7, 10, 5]) #checks out

### CHECK ASSERTIONS
#argmax(3) #checks out

### K):
def categorical(label, classes = 10):
    return [0 if x != label else 1 for x in range(classes)]

### CHECK EXAMPLE:
categorical(3) #checks out

### L)
test = linear_load('mnist_linear.weights')
images = read_image('train-images.idx3-ubyte')
labels = read_labels('train-labels.idx1-ubyte')
image_vector = image_to_vector(images[0])

import matrix_functions2 as M

def predict(network, image):
    A, b = network
    image = [image] #manual for now
    xA = M.multiply(image, A)
    dim_xA_rows, dim_xA_cols = M.dim(xA)
    dim_b_rows, dim_b_cols = M.dim(b)
    b = [b] #manual for now.
    xAb = M.add(xA, b)
    xAb_unlisted = xAb[0]
    return xAb_unlisted

### M)

def evaluate(network, images, labels):
    predictions = []
    cost = 0
    accuracy = 0
    for i in range(len(images)):
        image_vector = image_to_vector(images[i])
        prediction = predict(network, image_vector)
        prediction_label = argmax(prediction)
        cost += mean_square_error(prediction, categorical(labels[i]))
        if prediction_label == labels[i]:
            accuracy += 1
        predictions.append(prediction_label)
    return (predictions, cost/len(images), accuracy/len(images))


### N)
index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
predictions = [7, 2, 1, 3, 4, 5, 6, 6, 8, 9, 10, 11]
filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}
images = read_image(filename['images'])
filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = read_labels(filename['images'])
predictions = [7, 2, 1, 3, 4, 5, 6, 6, 8, 9, 10, 11]

## consider doing **kwargs instead ##

def plot_images_new(images, labels, index_list, predictions = labels):
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

plot_images_new(images, labels, index_list, predictions)

## O)

#Right now we assume input of 10 weights.
#Could be made more flexible.

network = linear_load('mnist_linear.weights')
A, b = network

#assumes
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

weights_plot(A)
### P)

'''
Create function create_batches(values, batch_size)
that partitions a list of values into batches of
size batch_size, except for the last batch, that can be smaller.
The list should be permuted before being cut into batches.
Example: create_batches(list(range(7)), 3)
should return [[3, 0, 1], [2, 5, 4], [6]].
'''

import random

def create_batches(values, batch_size):
    '''Using the random.shuffle function from the random module,
    this function partitions a list of values into random batches of
    length batch_size. The only exception is the last batch, which can be
    of a smaller length.
    Assumes that the input is a list and that batch_size is an integer.
    Returns a list of batches of values.
    '''
    values_list = []
    values_copy = values[:]
    random.shuffle(values_copy)
    current_batch = 0

    while current_batch < len(values_copy):
        current_batch += batch_size
        values_list.append(values_copy[current_batch-batch_size:current_batch])

    return values_list

l = create_batches(list(range(8)), 3)

### Q)
'''
Create a function update(network, images, labels)
that updates the network network = (A, b)
given a batch of n image vectors and corresponding
output labels (performs one step of a stochastical gradient
descend in the 784 * 10 + 10 = 7850 dimensional space
where all entries of A and b are considered to be variables).
For each input in the batch, we consider the tuple (x, a, y),
where x is the image vector, a = xA + b the current network's
output on input x, and y the corresponding categorical vector
for the label. The biases b and weights A are updated as
follows:
b j −= σ · (1 / n) · ∑ (x,a,y) 2 · ( a j − y j) / 10
A ij −= σ · (1 / n) · ∑ (x,a,y) x i · 2 · ( a j − y j) / 10
For this problem an appropriate value for the step size σ
of the gradient descend is σ = 0.1.
In the above equations 2 · (aj −yj) / 10
is the derivative of the cost function (mean squared error)
wrt. to the output aj, whereas xi · 2 · (aj − yj) / 10
is the derivative of the cost function w.r.t. to Aij — both
for a specific image (x, a, y).
'''

filename_test = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 't10k-labels.idx1-ubyte'}
filename_train = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}

labels = read_labels(filename_train['labels'])
images = read_image(filename_train['images'])
network = linear_load('mnist_linear.weights')

#something something batches, make cleaner.
#also consider what we actually need.
batches = create_batches(list(range(len(images))), 10)
image_batch = []
label_batch = []
for i in batches:
    one_img_batch = [images[j] for j in i]
    image_batch.append(one_img_batch)
    one_lab_batch = [labels[j] for j in i]
    label_batch.append(one_lab_batch)

def update(network, images, labels):
    batches = create_batches(list(range(len(images))), 10)



## ok
