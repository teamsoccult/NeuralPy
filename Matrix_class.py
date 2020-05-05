class Matrix:

    def __init__(self, L):
        self.matrix = L
        self.dim = dim(self.matrix)

    #consider
    def dim(V):
        if not type(V) == list:
            return []
        return [len(V)] + dim(V[0])

    def add(self, other):
        if self.dim != other.dim :
            raise ValueError(" The two matrices do not have the same dimensions.")
        rows, columns = self.dim
        A = self.gen_row()
        B = other.gen_row()
        C = [[] for i in range(rows)]
        for i in range(rows):
            for j in range(columns):
                C[i].append(next(A) + next(B))
        return C

    def sub(self, other):
        if self.dim != other.dim:
            raise ValueError("The two matrices do not have the same dimensions.")
        rows, columns = self.dim
        A = self.gen_row()
        B = other.gen_row()
        C = [[] for i in range(rows)]
        for i in range(rows):
            for j in range(columns):
                C[i].append(next(A) - next(B))
        return C

    def scalar_multiplication(self, scalar):
        generator_self = self.gen_row()
        rows, columns = self.dim
        C = [[] for i in range(rows)]
        for i in range(rows):
            for j in range(columns):
                C[i].append(next(generator_self)*scalar)
        return C


    def multiply(self, other):
        self_rows, self_columns = self.dim
        other_rows, other_columns = other.dim

        if self_columns != other_rows:
            raise ValueError('''The two matrices do not match for matrix multiplication.
        There must be the same number of rows in the first matrix as the number of columns in the second.''')

        C = [[] for i in range(self_rows)]

        sum_of_matrices = 0

        for m in range(self_rows):
            B_cols = other.gen_col()
            for i in range(self_columns):
                for j in range(other_rows):
                    sum_of_matrices +=  self.matrix[m][j] * next(B_cols)
                C[m].append(sum_of_matrices)
                sum_of_matrices = 0
        return C

    def transpose(self):
        A = self.gen_row()
        rows, columns = self.dim # maybe smart.
        C = [[] for i in range(columns)] # empty list
        for i in range(rows): #will have the opposite dimensionality.
            for j in range(columns):
                C[j].append(next(A))
        return(C)

    def gen_row(self):
        for i in self.matrix:
            for j in i:
                yield j

    def gen_col(self):
        for i in range(len(self.matrix[0])):
            for j in self.matrix:
                yield j[i]

##INITIALIZING:
list_matrix = [[1,2,3],[2,3,4],[2,3,5]]

list_matrix2 = [[4,3,2],[5,3,2],[1,2,4]]

### INSTANCES OF MATRIX:

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
