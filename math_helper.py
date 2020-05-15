'''
TASKS: H, I, J, K, L, M
'''
import read_write_helper as RW
### HELPER FUNCTIONS

def dim(S):
    dim_list = dim_recursive(S)
    if len(dim_list) < 2:
        dim_list.insert(0,1)
    return dim_list

def dim_recursive(S):
    if not type(S) == list:
        return []
    return [len(S)] + dim_recursive(S[0])

## Generators
def gen_row(S):
    for i in S:
        for j in i:
            yield j

def gen_col(S):
    for i in range(len(S[0])):
        for j in S:
            yield j[i]

### TASK H)

def add(S, O):
    if dim(S) != dim(O):
        raise ValueError("The two matrices do not have the same dimensions.")
    rows, columns = dim(S)
    A = gen_row(S)
    B = gen_row(O)
    C = [[] for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            C[i].append(next(A) + next(B))
    return C

def sub(S, O):
    if dim(S) != dim(O):
        raise ValueError("The two matrices do not have the same dimensions.")
    rows, columns = dim(S)
    A = gen_row(S)
    B = gen_row(O)
    C = [[] for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            C[i].append(next(A) - next(B))
    return C

def scalar_multiplication(S, scalar):
    generator_self = gen_row(S)
    rows, columns = dim(S)
    C = [[] for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            C[i].append(next(generator_self)*scalar)
    return C

def multiply(S, O):
    self_rows, self_columns = dim(S)
    other_rows, other_columns = dim(O)

    if self_columns != other_rows:
        raise ValueError('''The two matrices do not match for matrix multiplication.
    There must be the same number of rows in the first matrix as the number of columns in the second.''')

    C = [[] for i in range(self_rows)]

    sum_of_matrices = 0

    for m in range(self_rows):
        B_cols = gen_col(O)
        for i in range(other_columns):
            for j in range(self_columns):
                sum_of_matrices +=  S[m][j] * next(B_cols)
            C[m].append(sum_of_matrices)
            sum_of_matrices = 0
    return C

def transpose(S):
    A = gen_row(S)
    rows, columns = dim(S) # maybe smart.
    C = [[] for i in range(columns)] # empty list
    for i in range(rows): #will have the opposite dimensionality.
        for j in range(columns):
            C[j].append(next(A))
    return(C)

### TASK I)

#also assert that both should be of equal length
def mean_square_error(U, V):
    if not isinstance(U, list) or not isinstance(V, list):
        raise TypeError("Input must be lists.")
    vector_sum = 0
    for i in range(len(U)):
        vector_sum += (V[i]-U[i])**2
    return vector_sum/len(U)

### TASK J)

def argmax(V): ### inspired by https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
    if not isinstance(V, list):
        raise TypeError("Input must be a list.")

    return V.index(max(V))

### TASK K):
def categorical(label, classes = 10):
    return [0 if x != label else 1 for x in range(classes)]

### TASK L):
def predict(network, image):
    '''
    Description:
    Multiplies an image vector with the weights of a given network,
    and adds this product with the bias of the network.
    (is this correct?)
    This corresponds to the networks prediction of what the image is.

    ________

    Assumptions:
    Assumes that network is a nested list, with the sub-elements.
    The first element (A) should have the same number of rows that
    the image

    ________

    Returns:
    Returns a list of length equal to b (bias vector).
    say something more..

    ________

    Keyword arguments:
    image -- image vector (list) with of size [1, x] where
    x is the number of columns (so it is a row vector).
    network -- list with size [2, y, z] where y is the number
    of

    ________

    Examples:
    >>> predict([[[2,3],[2,2],[1,2],[1,2]],[2,3]], [1,2,4,0])
    [12, 18]

    '''
    A, b = network
    image = [image] #manual for now
    xA = multiply(image, A)
    dim_xA_rows, dim_xA_cols = dim(xA)
    dim_b_rows, dim_b_cols = dim(b)
    b = [b] #manual for now.
    xAb = add(xA, b)
    xAb_unlisted = xAb[0]
    return xAb_unlisted

### TASK M)

def evaluate(network, images, labels):
    predictions = []
    cost = 0
    accuracy = 0
    for i in range(len(images)):
        image_vector = RW.image_to_vector(images[i])
        prediction = predict(network, image_vector)
        prediction_label = argmax(prediction)
        cost += mean_square_error(prediction, categorical(labels[i]))
        if prediction_label == labels[i]:
            accuracy += 1
        predictions.append(prediction_label)
    return (predictions, cost/len(images), accuracy/len(images))

# ### Test:

# list_matrix = [[1,2,3], [4,5,6]]
# list_matrix2 = [[2,3],[4,5], [6,7]]
# list_matrix3 = [[2,3,4], [5, 6,7]]

# ### Dim:

# dim(list_matrix)
# dim(list_matrix2)
# dim([1,2,3])

# ### Add and sub:
# add(list_matrix, list_matrix3)
# sub(list_matrix3, list_matrix)

# ###Scalar Multiplication
# scalar_multiplication(list_matrix, 5)

# ### Matrix multiplication:

# multiply(list_matrix, list_matrix2)

# multiply(list_matrix2, list_matrix)

# multiply(list_matrix3, list_matrix)

# ### Transpose:

# transpose(list_matrix)
# transpose(list_matrix2)
