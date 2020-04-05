# MatrixCalculator
Python Module that contains functions for matrix operations and theoretical interpretations in Linear Algebra (ranging from basic matrix algebra to determinants, vector spaces, and eigenvalues/vectors)

# Usage
move the module MatrixOperations.py into your working directory and import it into the file of your choosing
```
import MatrixOperations as mo
```
# Module Functions
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
| swap | swaps two specified rows of a matrix |
| pivot | returns the index of the first pivot position |
| echelon | row reduces a given matrix to echelon form (upper triangular form) |
| reducedEchelon | row reduces a given matrix to reduced echelon form (RREF) |
| augment | combines a coeffecient matrix and an answer vector into an augmented matrix |
| submatrix | given an n * n matrix and a row and column index, returns a submatrix disregarding specified row/column |
| solve | solves the equation Ax = b for the x vector, where A is a given coeffecient matrix and b is a given answer vector |
| determinant | finds the determinant of a given matrix |
| recursiveDeterminant | finds the determinant of a given matrix recursively |
