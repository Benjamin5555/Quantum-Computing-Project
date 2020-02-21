"""Selection of classes that represent matricies and Vectors implemented using sparse matricies 


.. todo::
    * General:
        * Testing required on all functions
        * Improved error messages & Error checking
        * Formatting of latex in transpose method
    * SparseMatrix:
        * Support addition of a scalar value (i.e. scalar* identity matrix)
        * Different way of accesing an element of a matrix i.e. more efficient methods 
          also zero index?
    * SquareMatrix:
        * Different init i.e. using a sparse array?
    * Vector:
        * Different init i.e. using a sparse array?
Author(s):
    * Benjamin Carpenter(s1731178@ed.ac.uk)

"""
from scipy.sparse import csc_matrix, csr_matrix, kron
import numpy as np

class SparseMatrix(object):
    """A general representation of a sparse matrix with common functionality for matrices:
      Addition, subtraction, matrix multiplication and Equality checking 

        Relies upon the scipy sparse module
    Attributes:
        matrix: The actual stored matrix
    """

    def __init__(self, matrix):
        assert not isinstance(matrix[0][0],list) #Ensure only 2D list 
        self.matrix = csr_matrix(matrix)
        self.matrix.eliminate_zeros()

    def tensor_product(self,matrix):
        """Returns the kronker product of this  matrix with another, when applied to a vector 
        returns the tensor product a specific case of the kronker product
        Args:
            matrix:A sparse matrix acting as the right hand side of the product

        Returns:
            A sparse matrix representation of self \\(\otimes\\) matrixB
    
        .. todo:: Testing, better commenting and checking this is actually the correct operation
        """
        return kron(self.matrix, matrix.matrix)
    
    def dot(self,matrix):
        """dot/scalar product of two matrices

        Args:
            matrix:A matrix, the same dimensions as the current matrix that will to be dotted with
        Returns:
            The dot (scalar) product of two matrices A and B
        
        Raises:
            TypeError: on invalid maticie sizes 
        """
        return self.matrix.dot(matrix)

    def __str__(self):
        return str(np.array(self.matrix.toarray()))

    def __getitem__(self, index):
        return self.matrix.A[index]

    def __add__(self,matrix):
        """
        Args: 
            matrix: The matrix to be added to the current matrix

        Returns:
            The addition of the two matrices
        
        Raises:
            AssertionError: If the matrices are of an different size will raise an error

        """
        assert np.shape(self.matrix) == np.shape(matrix.matrix) #Ensure valid operation 
                                                                #i.e. same sized vectors 
        return type(self)(self.matrix + matrix.matrix) #type(self) means if called from a vector 
                                                       #child class will return a vector and not a
                                                       #general matrix

    def __sub__(self,matrix):
        """
        Args: 
            matrix: The matrix to be subtracted from the current matrix

        Returns:
            The subtraction of the two matrices
        
        Raises:
            AssertionError: If the matrices are of an different size will raise an error

        """
        assert np.shape(self.matrix) == np.shape(matrix.matrix) #Ensure valid operation 
                                                                #i.e. same sized vectors 
        return type(self)(self.matrix + (matrix.matrix*-1))#SEE __add__ for type(self) bit
    
    def __mul__(self,multiplier):
        """
        Args:
            multiplier: A scalar or matrix to multiply the current matrix by 

        Returns:
            The scalar or matrix multiple of the current matrix and multiplier
        
        Raises:

        """
        if(isinstance(multiplier,SparseMatrix)):
            return type(self)(self.matrix*multiplier.matrix)#SEE __add__ for type(self) bit
        else:
            return type(self)(self.matrix.multiply(multiplier))#SEE __add__ for type(self) bit


    def __eq__(self,matrix):
        if(self.matrix.nnz == matrix.matrix.nnz and \
           self.matrix.get_shape() == matrix.matrix.get_shape()): #Check number of items/shape is
                                                                  #the same as a quick initial check
            return all(self.matrix.data==matrix.matrix.data)\
                       and all(self.matrix.indices == matrix.matrix.indices)\
                       and all(self.matrix.indptr == matrix.matrix.indptr)
                       #Compare wheter non zero elements have same data
        else:
            return False

    def transpose(self):
        """
        Returns:
            Transpose of the given matrix \\(a_{ij}^T = a_{ji}\\)
        """
        return type(self)(self.matrix.transpose())#SEE __add__ for type(self) bit
        

    def conjugate(self):
        """
        Returns:
            Conjugate matrix of the given matrix i.e. \\([a_{ij}]^*=[a^*_{ij}]\\)
        """
        return type(self)(self.matrix.conjugate())#SEE __add__ for type(self) bit
        

    
    def conjugate_transpose(self):
        """
        Returns:
            Conjugate transpose of the given matrix i.e. \\(([a_{ij}]^*)^T = a^*_{ji}\\)
        """
        return type(self)(self.matrix.conjugate().transpose()) #SEE __add__ for type(self) bit
    
    def __pow__(self,exponent):
        """
        Returns:
            Original matrix raised to the argument power
        """
        NotImplemented
    
class SquareMatrix(SparseMatrix):
    """Sparse square matrix that provides basic functionallity

    Attributes:
        matrix: A scipy sparse matrix storing data represented by matrix 
    """
    def __init__(self,matrix):
        """Creates a sparse matrix of passed matrix 

        Args:
            matrix: Matrix i.e. 2D list, 2D numpy array, or (not implemented)sparse matrix of matrix
        
        Raises:
            AssertionError: On recieving an incorrect shaped matrix (i.e. non 2D square matrix)
        """
    
        assert np.shape(matrix)[1] == np.shape(matrix)[0] #Ensure NxN matrix (i.e. square)
        
        
        #assert not isinstance(matrix[0][0],list)          #Ensure only 2D list, 
        #test doesn't work well with sparse martircies, consider different test or total removal 

        self.matrix = csr_matrix(matrix)
        self.matrix.eliminate_zeros()

class Vector(SparseMatrix): 
    """Vector (Row or column) representation that uses scipy sparse class  
    
    Attributes:
        matrix: a sparse column vector containing the values of the vector
        dimension: The number of dimensions represented within the vector
        type: Whether the vector is a row or column (col) vector
    """

    def __init__(self,matrix):
        """
        Args:
            matrix: A row (e.g. `[1,0,1,0]' or column `[[1],[0],[1],[0]]' formatted list-like object
        Raises:
            ValueError: On recieving invalid shaped list-like object


        .. todo:: Override transpose and related functions to update 'type' attribute
        """
        shape = np.shape(matrix)
        
        if(len(shape) == 1):
            self.type = "row"
        
        elif(shape[1]==1): #Ensure is either column or row vector 
            self.type = "col"
        else:
            raise ValueError("Cannot construct a vector with shape ",shape)
        
        self.matrix = csc_matrix(matrix)
        self.matrix.eliminate_zeros()
        self.dimension = np.shape(matrix)
    
    def __getitem__(self,index): 
        if self.type == "row":#Access is different depending on vector type
            return self.matrix.A[0][index] #matrix stores as [[1,2,3,4]] so we do [0] first
        else:
            return self.matrix.A[index][0] #matrix stored as [[1],[2],[5]] so we do [0] second

    #def transpose(self):
    #    """
    #    Returns:
    #        Transpose of the given vector \\(a_{ij}^T = a_{ji}\\)
    #    """
    #    if (self.type == "c"):
    #        self.type = "r"
    #    else:
    #        self.type = "c"

    #    return super().transpose()
