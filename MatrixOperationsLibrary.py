import numpy as np

#-------------------------------------------------User Functions---------------------------------------------------#
def loadMatrix(filename):
    linelist = []
    with open(filename, 'r') as file:
        linelist = file.readlines()
    rows = len(linelist)
    cols = len(linelist[0].split())
    matrix = np.zeros(shape=(rows, cols))
    for i in range(rows):
        line = linelist[i].split()
        for j in range(cols):
            matrix[i][j] = line[j]
    print("matrix loaded ...")
    return matrix

def numCols(matrix):
    return len(matrix[0])

def numRows(matrix):
    return len(matrix)


def transpose(mat):
    rows = numRows(mat)
    cols = numCols(mat)
    matrix = np.zeros(shape=(cols, rows))
    for i in range(rows):
        for j in range(cols):
            matrix[j][i] = mat[i][j]
    return matrix


def isVector(matrix):
    rows = numRows(matrix)
    cols = numCols(matrix)
    return (cols == 1 or rows == 1)


def isSquare(matrix):
    return numCols(matrix) == numRows(matrix)


def identity(dim):
    matrix = np.zeros(shape=(dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                matrix[i][j] = 1
    return matrix


def add(mat1,mat2):
    if numRows(mat1) != numRows(mat2) or numCols(mat1) != numCols(mat2):
        print( "ERROR: operation not possible")
    else:
        rows = numRows(mat1)
        cols = numCols(mat1)
        matrix = np.zeros(shape=(rows, cols))
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = mat1[i][j]+mat2[i][j]
        return matrix


def subtract(mat1, mat2):
    if numRows(mat1) != numRows(mat2) or numCols(mat1) != numCols(mat2):
        print("ERROR: operation not possible")
    else:
        rows = numRows(mat1)
        cols = numCols(mat1)
        matrix = np.zeros(shape=(rows, cols))
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = mat1[i][j]-mat2[i][j]
        return matrix
            
def scale(mat,scale):
    rows = numRows(mat)
    cols = numCols(mat)
    matrix = np.zeros(shape=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            matrix[i][j] = scale*mat[i][j]
    return matrix


def dot(vector1,vector2):
    rows1 = numRows(vector1)
    rows2 = numRows(vector2)
    cols1 = numCols(vector1)
    cols2 = numCols(vector2)
    ans = 0
    if(cols1 == 1):
        vector1 = transpose(vector1)
    if(rows2==1):
        vector2 = transpose(vector2)
    if(numCols(vector1)!=numRows(vector2)):
        return "ERROR: vectors must be in the same dimension (same # of entries)"
    if(isVector(vector1) and isVector(vector2)):
        for i in range(numCols(vector1)):
            ans+=vector1[0][i]*vector2[i][0]
    else:
        return "ERROR: must use a vector (single column/row matrix) to use the dot() function"
    return ans

def cross(vector1,vector2):
    rows1 = numRows(vector1)
    rows2 = numRows(vector2)
    cols1 = numCols(vector1)
    cols2 = numCols(vector2)
    ans = np.zeros(shape=(3, 1))
    if(cols1 == 1):
        vector1 = transpose(vector1)
    if(rows2==1):
        vector2 = transpose(vector2)
    if(numCols(vector1)!= 3 or numRows(vector2)!=3):
        return "ERROR: cross product is defined as an operation in three dimensional space, therefore the vectors must have three entries each"
    if(isVector(vector1) and isVector(vector2)):
        print(ans)
        print(vector1)
        print(vector2)
        ans[0][0] = vector1[0][1]*vector2[2][0]-vector1[0][2]*vector2[1][0]
        ans[1][0] = vector1[0][2]*vector2[0][0]-vector1[0][0]*vector2[2][0]
        ans[2][0] = vector1[0][0]*vector2[1][0]-vector1[0][1]*vector2[0][0]
    else:
        return "ERROR: must use a 3D vector (single column/row matrix with three entries) for the cross() function"
    return ans

def multiply(mat1,mat2):
    rows1 = numRows(mat1)
    rows2 = numRows(mat2)
    cols1 = numCols(mat1)
    cols2 = numCols(mat2)
    if cols1 != rows2:
        return "ERROR: to be multiplied, the # of columns in matrix 1 must be equal to the # of rows in matrix 2"
    ans = np.zeros(shape=(rows1,cols2))
    for i in range(rows1):
        for j in range(cols2):
            for n in range(cols1):
                ans[i][j] += mat1[i][n]*mat2[n][j]
    return ans

def elementPower(matrix, power):
    rows = numRows(matrix)
    cols = numCols(matrix)
    ans = np.zeros(shape=(rows,cols))
    for i in range(rows):
        for j in range(cols):
            ans[i][j] = matrix[i][j]**power
    return ans

def power(matrix,power):
    rows = numRows(matrix)
    cols = numCols(matrix)
    ans = np.zeros(shape=(rows,cols))
    if not isSquare(matrix):
        return "ERROR: matrix must be square to use this function"
    for n in range(power):
        ans*=matrix
    return ans

def swap(matrix,row1,row2):
    ans = copyMatrix(matrix)
    ans[[row1,row2]] = ans[[row2,row1]]
    return ans

def pivot(matrix, index):
    i = index
    j = index
    rows = numRows(matrix)
    cols = numCols(matrix)
    while j<cols and matrix[i][j] == 0:
        i+=1
        if i>=rows:
            i = index
            j += 1
    return (i,j) if j<cols else (-1,-1)


def upperTriangle(matrix):
    # print("----------------EF/UT-----------------")
    rows = numRows(matrix)
    cols = numCols(matrix)
    swaps = 0
    mat = copyMatrix(matrix)
    for n in range(min(rows, cols)):
        pivoti,pivotj = pivot(mat, n)
        if (pivoti == -1 or pivotj == -1):
            break
        if mat[pivoti][pivotj] < 0:
            for x in range(cols):
                mat[pivoti][x] *= -1
        if (n!=pivoti):
            swaps += 1
            mat = swap(mat, n, pivoti)
        for i in range(pivoti+1, rows):
            ratio = mat[i][pivotj]/mat[n][pivotj]
            for j in range(pivotj, cols):
                mat[i][j] -= ratio*mat[n][j]
        # print(mat)
        # print()
    return mat,swaps

def echelon(matrix):
    mat = upperTriangle(matrix)[0]
    rows = numRows(mat)
    cols = numCols(mat)
    for i in range(rows):
        j = 0
        while j<cols and mat[i][j]==0:
            j+=1
        divisor = mat[i][j] if j<cols else 1
        for j in range(cols):
            if divisor != 0:
                mat[i][j] /= divisor
            mat[i][j] = 0 if mat[i][j] == -0 else mat[i][j]
    # print(mat)
    # print()
    return mat


def reducedEchelon(matrix):
    rows = numRows(matrix)
    cols = numCols(matrix)
    mat = echelon(matrix)
    # print("----------------RREF-----------------")
    for n in range(min(rows,cols)):
        pivoti,pivotj = pivot(mat, n)
        if pivotj >= cols:
            break
        if mat[n][pivotj] == 1:
            for i in range(0,pivoti):
                ratio = mat[i][pivotj]/mat[n][pivotj]
                for j in range(pivotj, cols):
                    mat[i][j] -= ratio*mat[n][j]
        # print(mat)
        # print()
    for i in range(rows):
        for j in range(cols):
            mat[i][j] = 0 if mat[i][j] == -0 else mat[i][j]
    return mat


def submatrix(matrix, indexi, indexj):
    rows = numRows(matrix)
    cols = numCols(matrix)
    icount = 0
    jcount = 0
    submatrix = np.zeros(shape=(rows-1, cols-1))
    for i in range(rows):
        if (i == indexi):
            icount = 1
            continue
        for j in range(cols):
            if (j == indexj):
                jcount = 1
                continue
            submatrix[i-icount][j-jcount] = matrix[i][j]
        jcount = 0
    return submatrix

def augment(matrix1, matrix2):
    rows1 = numRows(matrix1)
    cols1 = numCols(matrix1)
    rows2 = numRows(matrix2)
    cols2 = numCols(matrix2)
    if rows1 != rows2:
        return "ERROR: answer does not have the right number of entries to be augmented with the given matrix"
    augmented = np.zeros(shape=(rows1, cols1+cols2))
    for i in range(rows1):
        for j in range(cols1+cols2):
            if j<cols1:
                augmented[i][j] = matrix1[i][j]
            else:
                augmented[i][j] = matrix2[i][j-cols1]
    return augmented

def linearDependence(matrix):
    cols = numCols(matrix)
    totalpivots = totalPivots(reducedEchelon(matrix))
    if totalpivots == cols:
        print("The columns of the matrix are linearly independent as there are pivot positions in each column")
        return False
    else:
        print("The columns of the matrix are linearly dependent as each column does not have a pivot position, \ntherefore there are free variables")
        return True
    

def determinant(matrix):
    rows = numRows(matrix)
    cols = numCols(matrix)
    if(not isSquare(matrix)):
        return "ERROR: must use a square matrix for the determinant() function"
    mat,interchanges = upperTriangle(matrix)
    ans = 1
    for n in range(min(rows,cols)):
        ans *= mat[n][n]
    return ans if interchanges%2==0 else ans*(-1)

def recursiveDeterminant(matrix):
    ans = 0
    if(not isSquare(matrix)):
        return "ERROR: must use a square matrix for the determinant() function"
    if(numRows(matrix)==2):
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    else:
        for j in range(numCols(matrix)):
            ans += (-1)**(j)*matrix[0][j]*recursiveDeterminant(submatrix(matrix,0,j))
    return ans    


def inverse(matrix):
    rows = numRows(matrix)
    cols = numCols(matrix)
    if not isSquare(matrix):
        return "ERROR: matrix must be square to be invertible\n"
    if determinant(matrix) == 0:
        return "ERROR: determinant is zero, therefore this matrix is not invertible\n"
    identitymatrix = identity(rows)
    augmented = augment(matrix, identitymatrix)
    reduced = reducedEchelon(augmented)
    ans = np.zeros(shape=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            ans[i][j] = reduced[i][j+cols]
    return ans


def gaussianSolve(A,b):
    rows = numRows(A)
    colsA = numCols(A)
    if rows != numRows(b) or numCols(b) != 1:
        return "ERROR: b must be a column vector and the # of rows in matrix A and vector b should be the same"
    augmentmatrix = reducedEchelon(augment(A,b))
    reducedA = np.zeros(shape=(rows,colsA))
    identitymatrix = identity(rows)
    xvector = np.zeros(shape=(colsA,1))
    totalpivots = totalPivots(reducedA)
    for i in range(rows):
        if augmentmatrix[i][i] == 1:
            xvector[i][0] = augmentmatrix[i][colsA] 
        for j in range(colsA):
            reducedA[i][j] = augmentmatrix[i][j]
    for i in range(rows):
        if np.array_equal(reducedA[i],[0]*colsA) and augmentmatrix[i][colsA-1] != 0: # testing if system is INCONSISTENT
            return "The given system Ax = b is inconsistent (there are no solutions for the x vector)"
    if np.array_equal(identitymatrix,reducedA): # testing if the system is CONSISTENT and UNIQUE
        return xvector
    else: # the system is CONSISTENT but NOT UNIQUE (infinite solutions due to free variables)
        freevars = {"x"+str(n+1):[0.0]*colsA for n in range(colsA)}
        def parametricVectorForm():
            key = "x"+str(j+1)
            freevars[key][j] = 1.0
            for i in range(rows):
                if i != j:
                    freevars[key][i] = (-1) * \
                        augmentmatrix[i][j] if augmentmatrix[i][j] != 0 else 0.0
        for j in range(colsA):
            if j >= rows:
                parametricVectorForm()
            elif augmentmatrix[j][j] != 1:
                parametricVectorForm()
    xvector = transpose(xvector)[0]
    xvectorStr = "\033[1m" + "x = " + "\033[0m" + str(xvector) if not np.array_equal(xvector, [0]*colsA) else ""
    for j in range(colsA):
        xkey = "x"+str(j+1)
        if freevars[xkey] != [0]*colsA:
            xvectorStr += " + "+str(freevars[xkey])+" * "+xkey
        else:
            del freevars[xkey]
    return xvectorStr


def inverseSolve(A,b):
    rows = numRows(A)
    colsA = numCols(A)
    if rows != numRows(b) or numCols(b) != 1:
        return "ERROR: b must be a column vector and the # of rows in matrix A and vector b should be the same\n"
    if not isSquare(A):
        return "ERROR: matrix A must be square to be invertible\n"
    if determinant(A) == 0:
        return "ERROR: to use inverseSolve, matrix A must be invertible\n"
    inverseA = inverse(A)
    return multiply(inverseA,b)


def cramersSolve(A,b):
    rows = numRows(A)
    colsA = numCols(A)
    xvector = np.zeros(shape=(colsA, 1))
    detA = determinant(A)
    if detA == 0:
        return "ERROR: to use cramersSolve, matrix A must be invertible\n"
    if rows != numRows(b) or numCols(b) != 1:
        return "ERROR: b must be a column vector and the # of rows in matrix A and vector b should be the same\n"
    for j in range(colsA):
        detAjb = determinant(replaceCol(A,j,b))
        xvector[j] = detAjb/detA
        if xvector[j] == -0: xvector[j] = 0
    return xvector


# def dimension(matrix):



#-------------------------------------------------Non-user Functions---------------------------------------------------#

def copyMatrix(matrix):
    rows = numRows(matrix)
    cols = numCols(matrix)
    new = np.zeros(shape=(rows,cols))
    for i in range(rows):
        for j in range(cols):
            new[i][j] = matrix[i][j]
    return new    

def totalPivots(matrix):
    rows = numRows(matrix)
    cols = numCols(matrix)
    pivots = 0
    for n in range(min(rows, cols)):
        pivots += 1 if pivot(matrix, n) != (-1, -1) else 0
    return pivots

def replaceCol(matrix,index,col1):
    rows = numRows(matrix)
    if len(col1) != rows:
        return "ERROR: the given column and matrix do not have the same number of rows"
    new = copyMatrix(matrix)
    for i in range(rows):
        new[i][index] = col1[i][0]
    return new
