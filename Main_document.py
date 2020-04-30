import struct

### TASK B)

filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}

def read_labels(filename):
    with open(filename, 'rb') as f:
        #zero, data_type, dims = struct.unpack('>H2B', f.read(4))
        #shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        f.seek(2) #start after the 2 zeroes
        magic = struct.unpack('>H',f.read(2)) #magic number as hex digit
        if magic[0] == 2049:
            print('Hurray! The Magic Number is 2049!')
        f.seek(6) #offset 
        test = struct.unpack('>H', f.read(2)) #number of items
        f.seek(8)
        labels = struct.unpack('>10000B', f.read(10000))
        return list(labels)

labels = read_labels(filename['images'])

labels

### TASK C)

filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}


def read_image(filename):
    with open(filename, 'rb') as f:
        #zero, data_type, dims = struct.unpack('>H2B', f.read(4))
        #shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        f.seek(2) #start after the 2 zeroes
        magic = struct.unpack('>H',f.read(2)) #magic number as hex digit
        if magic[0] == 2051:
            print('Hurray! The Magic Number is 2049!') 
        noIm = struct.unpack('>HH', f.read(4)) #number of items
        noR = struct.unpack('>HH', f.read(4))
        noC = struct.unpack('>HH', f.read(4))
        labels = struct.unpack('>10000B', f.read(10000))
        return list(labels)


with open(filename['images'], 'rb') as f:
    #zero, data_type, dims = struct.unpack('>H2B', f.read(4))
    #shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    f.seek(2) #start after the 2 zeroes
    magic = struct.unpack('>H',f.read(2)) #magic number as hex digit
    if magic[0] == 2051:
        print('Hurray! The Magic Number is 2049!') 
    f.seek(6)
    noIm = struct.unpack('>H', f.read(2)) #number of items
    f.seek(10)
    noR = struct.unpack('>H', f.read(2))
    f.seek(14)
    noC = struct.unpack('>H', f.read(2))
    #noD = noIm[0] * noR[0] * noC[0]

    images = list()

    for i in range(noIm[0]):
        image = list()
        for j in range(noR[0]):
            row = list(struct.unpack(">28B", f.read(28)))
            image.append(row)
        images.append(image)


import matplotlib.pyplot as plt

### DIFFERENT APPROACH: (INDEXES)
# should probably be done using map instead (cater it to 1 picture instead)

def plot_images(images, labels, indexes = [0]):
    fig, axs = plt.subplots(1, len(indexes))
    for i in range(len(indexes)):
        axs[i].imshow(images[indexes[i]], cmap = "binary")
        axs[i].axis("off")
        axs[i].set_title(labels[indexes[i]])
    plt.show()

plot_images(images, labels, [0, 4, 5, 6, 8, 14, 1, 56, 32, 123, 12])

### F):

import json

filename = "mnist_linear.weights"

def linear_load(filename):
    if filename:
        with open(filename, "r") as f:
            json_string = json.load(f)
    else:
        print("Cannot find file in the directory. \nPlease check the filename and the pathing to said filename.")
    return json_string

json_string = linear_load(filename)

### NEEDS CODE FROM VICTOR:

#def linear_save(filename, network):


### G):
def image_to_vector(image):
    '''Standardize image to be a single list (image vector).
    Assumes that the values of the image is between [0, 255].
    Returns a list of floats between [0,1]. '''
    return [(item)/(255) for sublist in image for item in sublist]



### H)

### CLASS OF MATRIX

class Matrix:

    def __init__(self, L):
        self.matrix = L[:]

    def add(self, other):
        if len(self.matrix) == len(other.matrix) and len(self.matrix[0]) == len(other.matrix[0]):
            A = self.gen_row()
            B = other.gen_row()
            C = [[] for i in range(len(self.matrix))]
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[0])):
                    C[i].append(next(A) + next(B))  
        else:
            raise ValueError(" The two matrices do not have the same dimensions.")
        return C
    
    def sub(self, other):
        if len(self.matrix) == len(other.matrix) and len(self.matrix[0]) == len(other.matrix[0]):
            A = self.gen_row()
            B = other.gen_row()
            C = [[] for i in range(len(self.matrix))]
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[0])):
                    C[i].append(next(A) - next(B))  
        else:
            raise ValueError(" The two matrices do not have the same dimensions.")
        return C

    def scalar_multiplication(self, scalar):
        generator_self = self.gen_row()
        C = [[] for i in range(len(self.matrix))]
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                C[i].append(next(generator_self)*scalar)
        return C


    def multiply(self, other):

        if len(self.matrix[0]) != len(other.matrix):
            raise "LengthError: The two matrices do not match for matrix multiplication. There must be the same number of rows in the first matrix as the number of columns in the second."
        
        C = [[] for i in range(len(self.matrix))]

        sum_of_matrices = 0

        for m in range(len(self.matrix)):
            B_cols = other.gen_col()
            for i in range(len(self.matrix)):

                for j in range(len(other.matrix[0])):
                    sum_of_matrices +=  self.matrix[m][j] * next(B_cols)
                C[m].append(sum_of_matrices)
                sum_of_matrices = 0

        return C
    
    def gen_row(self):
        for i in self.matrix:
            for j in i:
                yield j 
    
    def gen_col(self):
        for i in range(len(self.matrix[0])):
            for j in self.matrix:
                yield j[i] 


### TESTING THE MATRIX:

##INITIALIZING:
list_matrix = [[1,2,3],[2,3,4],[2,3,5]]

list_matrix2 = [[4,3,2],[5,3,2],[1,2,4]]

matrix1 = Matrix(list_matrix)
matrix2 = Matrix(list_matrix2)

### adding:

matrix1.add(matrix2)

### subtracting:

matrix1.sub(matrix2)

### scalar multi:
matrix1.scalar_multiplication(3)

### matrix mult:
matrix1.multiply(matrix2)

### I):

def mean_square_error(U, V):
    if not isinstance(U, list) or not isinstance(V, list):
        raise TypeError("Input must be lists.")
    vector_sum = 0
    for i in range(len(U)):
        vector_sum += (V[i]-U[i])**2
        print(vector_sum)
    return vector_sum/len(U)

### CHECK EXAMPLE:
mean_square_error([1,2,3,4], [3,1,3,2]) #checks out 

### CHECK ASSERTIONS
mean_square_error([1,2,3,4], 5) #checks out

### J):

V = [1,2,3,4]

def argmax(V): ### inspired by https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
    if not isinstance(V, list):
        raise TypeError("Input must be a list.")
    
    return V.index(max(V))

#CHECK EXAMPLE:
argmax([6, 2, 7, 10, 5]) #checks out

### CHECK ASSERTIONS
argmax(3) #checks out

### K):
def categorical(label, classes = 10):
    return [0 if x != label else 1 for x in range(classes)]

### CHECK EXAMPLE:
categorical(3) #checks out

