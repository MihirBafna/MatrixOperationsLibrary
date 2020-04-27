# MatrixOperationsLibrary

Python Module that contains functions for matrix operations and theoretical interpretations in Linear Algebra (ranging from basic matrix algebra to determinants, vector spaces, and eigenvalues/vectors)

## Usage

move the module MatrixOperations.py into your working directory and import it into the file of your choosing

```python
import MatrixOperationsLibrary as mol
```

## Module Functions

Here is the list of functions that the module currently contains:

| Function | Description |
| --- | --- |
| loadMatrix | reads matrix in from file and returns it as a 2D array |
| numCols | returns the number of columns for a specified matrix |
| numRows | returns the number of rows for a specified matrix |
| transpose | returns a matrix with the rows and columns swapped |
| isVector | returns whether or not the given matrix is a vector (1 dimensional) |
| isSquare | returns whether not the given matrix is square |
| identity | returns an identity matrix of specified size |
| add | adds two matrices |
| subtract | subtracts two matrices |
| scale | multiplies a given matrix with a scalar value |
| dot | dot product of two vectors |
| cross | cross product of two vectors |
| multiply | matrix product of two matrices |
| elementPower | raises each element of a matrix to a specified power |
| power | raises a given matrix to the given power (multiplying the matrix by itself n times where n is the given power) |
| swap | swaps two specified rows of a matrix |
| pivot | returns the indices of the first pivot position after ths given index |
| upperTriangle | returns a tuple of the given matrix in upper triangular form and the number of row interchanges |
| echelon | row reduces a given matrix to echelon form |
| reducedEchelon | row reduces a given matrix to reduced row echelon form (RREF) |
| augment | combines a coeffecient matrix and an answer vector into an augmented matrix |
| linearDependence | returns whether or not the columns of the given matrix are linearly dependent (boolean) |
| submatrix | given an n by n matrix and a row and column index, returns a n-1 by n-1 submatrix disregarding the specified row/column |
| determinant | finds the determinant of a given matrix |
| recursiveDeterminant | finds the determinant of a given matrix recursively |
| inverse | returns the inverse of a given matrix |
| gaussianSolve | using Gaussian elimination, solves the equation **Ax = b** for the **x** vector, where **A** is a given coeffecient matrix and **b** is a given answer vector. This function will determine if a system is inconsistent or return a dictionary where each key (const, x1, x1, x2, etc.) will have a vector as its value (this represents the Parametric Vector Form of the solution).  |
| inverseSolve | using the property of matrix invertability, solves the equation **Ax = b** for the **x** vector, where **A** is a given coeffecient matrix and **b** is a given answer vector |
| cramersSolve | using Cramer's Rule, solves the equation **Ax = b** for the **x** vector, where **A** is a given coeffecient matrix and **b** is a given answer vector |
| nullSpace | returns the null space of a given matrix A as a dictionary (solution to Ax=0 using gaussianSolve()) |
| colSpace | return the column space of a given matrix A as a dictionary |
