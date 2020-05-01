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