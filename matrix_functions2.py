### HELPER FUNCTIONS

#### Rows and columns might be reverse - back to the original concept of [rows][columns] instead of [columns][rows].

## Dimensions

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

### MAIN FUNCTIONS
def add(S, O):
    if dim(S) != dim(O):
        raise ValueError("The two matrices do not have the same dimensions.")
    rows, columns = dim(S)

    if rows == 1:
        C = []
        for j in range(columns):
            C.append(S[j] + O[j])
        return C

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

    if rows == 1:
        C = []
        for j in range(columns):
            C.append(S[j] - O[j])
        return C

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

    if self_rows == 1:
        C = []
        B_cols = gen_col(O)
        for i in range(other_columns):
            for j in S:
                sum_of_matrices += j * next(B_cols)
            C.append(sum_of_matrices)
            sum_of_matrices = 0
        return C

    for m in range(self_rows):
        B_cols = gen_col(O)
        for i in range(other_columns):
            for j in range(other_rows):
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

