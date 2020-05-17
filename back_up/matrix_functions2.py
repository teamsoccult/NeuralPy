### HELPER FUNCTIONS

#### Things to remember for the report:
'''
- decorator show which also became recursive
- matrix class
- [rows][columns] or [columns][row]
- list of lists necessary for the indexing
'''


## Dimensions
def dim_recursive(S):
    '''
    Description: 
    
    Recursive function to calculate the dimensions of 
    a list. 
    Assumes that the input is a list, and that the sublists
    are of equal lengths. 
    Returns a list of the dimensions of S. 

    ________
    Arguments:

    S = a list or list of lists
    ________  
    Examples:

    dim_recursive([[1,2,3], [2,3,4]]) = [2, 3].

    '''
    if not type(S) == list:
        return []
    return [len(S)] + dim_recursive(S[0])

help(dim_recursive)

def dim(S):
    '''
    Description:

    Using the dim_recursive(function), this function ensures
    that the result will always have 2 values.
    Assumes that the input is a list.
    Returns a list of the length of the dimensions. 
    ________
    Arguments:

    S = a list or list of lists
    ________
    Examples:

    dim([[1, 2, 3], [2, 3, 4]]) = [2, 3]

    dim([1,2,3]) = [1,3]

    '''
    dim_list = dim_recursive(S)
    if len(dim_list) < 2:
        dim_list.insert(0,1)
    return dim_list

help(dim)

## Generators
def gen_col(S):
    '''
    Description:

    Generator function which iterates through the columns 
    of a list of lists. 
    Assumes that the input is list of lists, with 2 dimensions. 
    Yields the element S[r][c], where r is iterated over first.
    
    ________
    Arguments:
    
    S = list of lists with 2 dimensions
    ________
    Examples:

    gen_columns = gen_col([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

    next(gen_columns) = 1
    next(gen_columns) = 3
    next(gen_columns) = 5

    '''
    for i in S:
        for j in i:
            yield j

help(gen_col)

def gen_row(S):
    '''
    Description:

    Generator function which iterates through the rows 
    of a list of lists. 
    Assumes that the input is list of lists with 2 dimensions, and that
    the length of each sublist is equal.
    Yields the element S[r][c], where c is iterated over first.

    ________
    Arguments:
    
    S = list of lists with 2 dimensions
    ________
    Examples:

    gen_rows = gen_row([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

    next(gen_rows) = 1
    next(gen_rows) = 2
    next(gen_rows) = 3

    '''
    for i in range(len(S[0])):
        for j in S:
            yield j[i]

### MAIN FUNCTIONS
def add(S, O):
    '''
    Description:

    Adds a matrix to another matrix. Raises a ValueError 
    if the matrices do not have the same dimensions.
    Corresponding to S + O.
    Assumes that both matrices have (1) the same dimensions and (2) 2 dimensions.
    Returns a list of lists, corresponding to the result of the addition 
    of the two matrices.

    ________
    Arguments:
    
    S = list of lists with 2 dimensions
    O = list of lists with 2 dimensions 
    ________
    Examples:

    add([[1, 2, 3], [3, 4, 5], [5,6,7]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) = [[2, 4, 6], [7, 9, 11], [12, 14, 16]]

    '''
    if dim(S) != dim(O):
        raise ValueError("The two matrices do not have the same dimensions.")
    columns, rows = dim(S)
    A = gen_col(S)
    B = gen_col(O)
    C = [[] for i in range(columns)]
    for i in range(columns):
        for j in range(rows):
            C[i].append(next(A) + next(B))
    return C

def sub(S, O):
    '''
    Description:

    Subtracts a matrix from another matrix. Raises a ValueError 
    if the matrices do not have the same dimensions. Corresponds to S - O.
    Assumes that both matrices have (1) the same dimensions and (2) 2 dimensions.
    Returns a list of lists, corresponding to the result of the subtraction 
    of the two matrices.

    ________
    Arguments:
    
    S = list of lists with 2 dimensions
    O = list of lists with 2 dimensions 
    ________
    Examples:

    sub([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [3, 4, 5], [5,6,7]]) = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    '''
    if dim(S) != dim(O):
        raise ValueError("The two matrices do not have the same dimensions.")
    columns, rows = dim(S)
    A = gen_col(S)
    B = gen_col(O)
    C = [[] for i in range(columns)]
    for i in range(columns):
        for j in range(rows):
            C[i].append(next(A) - next(B))
    return C

def scalar_multiplication(S, scalar):
    '''
    Description:

    Multiplies the matrix with a scalar. Corresponds to S * scalar.
    Assumes that S is a list of lists with 2 dimensions, and that
    scalar is either an integer or a float.
    Returns a list of lists, corresponding to the result of the subtraction 
    of the two matrices.

    ________
    Arguments:
    
    S = list of lists with 2 dimensions
    scalar = integer / float 
    ________
    Examples:

    scalar_multiplication([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2) = [[2, 4, 6], [8, 10, 12], [14, 16, 18]]

    '''
    if isinstance(scalar, int) or isinstance(scalar, float):
        generator_self = gen_col(S)
        columns, rows = dim(S)
        C = [[] for i in range(columns)]
        for i in range(columns):
            for j in range(rows):
                C[i].append(next(generator_self)*scalar)
        return C
    else:
        raise ValueError("Scalar must be either of type integer or type float.")

def multiply(S, O):
    '''
    Description:

    Multiplies the matrix with another matrix. Corresponds to S * O.
    Raises a ValueError if the amount of columns of the first matrix
    
    Assumes that S and O are lists of lists with 2 dimensions.
    
    Returns a list of lists, corresponding to the result of the subtraction 
    of the two matrices.

    ________
    Arguments:
    
    S = list of lists with 2 dimensions
    scalar = integer / float 
    ________
    Examples:

    scalar_multiplication([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2) = [[2, 4, 6], [8, 10, 12], [14, 16, 18]]

    '''
    self_columns, self_rows = dim(S)
    other_columns, other_rows = dim(O)

    if self_columns != other_rows:
        raise ValueError('''The two matrices do not match for matrix multiplication.
    There must be the same number of rows in the first matrix as the number of columns in the second.''')

    C = [[] for i in range(self_rows)]

    sum_of_matrices = 0

    for m in range(other_columns):
        B_cols = gen_col(O)
        for i in range(self_rows):
            for j in range(other_rows):
                sum_of_matrices +=  S[j][m] * next(B_cols)
            C[i].append(sum_of_matrices)
            sum_of_matrices = 0
    return C

def transpose(S):
    A = gen_col(S)
    columns, rows = dim(S) # maybe smart.
    C = [[] for i in range(rows)] # empty list
    for i in range(columns): #will have the opposite dimensionality.
        for j in range(rows):
            C[j].append(next(A))
    return(C)
