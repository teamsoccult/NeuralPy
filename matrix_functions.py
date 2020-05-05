### HELPER FUNCTIONS

## Dimensions
def dim(S):
    if not type(S) == list:
        return []
    return [len(S)] + dim(S[0])

## Generators
def gen_col(S):
    for i in S:
        for j in i:
            yield j

def gen_row(S):
    for i in range(len(S[0])):
        for j in S:
            yield j[i]

### MAIN FUNCTIONS
def add(S, O):
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
    generator_self = gen_col(S)
    columns, rows = dim(S)
    C = [[] for i in range(columns)]
    for i in range(columns):
        for j in range(rows):
            C[i].append(next(generator_self)*scalar)
    return C

def multiply(S, O):
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
    A = gen_row(S)
    rows, columns = dim(S) # maybe smart.
    C = [[] for i in range(columns)] # empty list
    for i in range(rows): #will have the opposite dimensionality.
        for j in range(columns):
            C[j].append(next(A))
    return(C)
