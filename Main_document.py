import struct as st

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

import matplotlib.pyplot as plt

### D):

### DIFFERENT APPROACH: (INDEXES)
# should probably be done using map instead (cater it to 1 picture instead)

def plot_images(images, labels, indexes = [0]):
    fig, axs = plt.subplots(1, len(indexes))
    for i in range(len(indexes)):
        axs[i].imshow(images[indexes[i]], cmap = "binary")
        axs[i].axis("off")
        axs[i].set_title(labels[indexes[i]])
    plt.show()

plot_images(images, labels, [0, 4, 5, 6, 8])

#def plot_image(image, label):
   # plt.imshow(image, cmap = "binary")
    #plt.axis("off")
    #plt.title(label)
    #return plt.show()

#map(plot_image, (images[1:2], labels[1:2]))

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
    index_list = []
    values_copy = list(enumerate(values[:]))
    random.shuffle(values_copy)
    current_batch = 0
    indexes, values = zip(*return_list)

    while current_batch < len(values_copy):
        current_batch += batch_size
        values_list.append(values[current_batch-batch_size:current_batch])

    return (indexes, values)

indices, l = create_batches(list(range(8)), 3)

list(indices)

l

indices, l

l = [4, 8, 15, 16, 23, 42]
x = list(enumerate(l))
random.shuffle(x)
indices, l = zip(*x)

indices, l


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
test_index = [image_to_vector(x) for x in images[0:10]]

testing = create_batches(test_index, 3)



x = testing[0][1]
a = 

