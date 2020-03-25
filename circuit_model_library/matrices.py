"""Selection of classes that represent matricies and Vectors implemented using sparse matricies 


.. todo::
    * General:
        * Testing required on all functions
        * Improved error messages & Error checking
        * Formatting of latex in transpose method
        * Compare different sparse matrix implementations to see which might be quicker
    * SparseMatrix:
        * Support addition of a scalar value (i.e. scalar* identity matrix)
        * Different way of accesing an element of a matrix i.e. more efficient methods 
          also zero index?
    
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
        matrix: The represented matrix
        shape: The dimensions of the matrix represented
    """

    def __init__(self, matrix):
        self.matrix = csr_matrix(matrix) #Define the represented matrix in sparse form
        self.matrix.eliminate_zeros()    #Eliminate any explicit zeros 
        self.shape = self.matrix.shape   #Define a shape attribute of the matrix

    def dot(self,matrix):
        """dot product of two matrices

        Args:
            matrix:A matrix like object, that will to be dotted with
        Returns:
            The dot product of two matrices 
        
        Raises:
            AssertionError: Where the object being dotted with is not matrix like 
        """
        if (isinstance(matrix,SparseMatrix)):
            return type(self)(self.matrix.dot(matrix.matrix))

    def tensor_product(self,matrix):
        """Returns the kronker product of this  matrix with another, when applied to a vector 
        returns the tensor product a specific case of the kronker product
        Args:
            matrix:A sparse matrix acting as the right hand side of the product

        Returns:
            A sparse matrix representation of self \\(\otimes\\) matrixB
    
        .. todo:: 
            * Testing, better commenting and checking this is actually the correct operation
            * Implement own tensor product function or explain how kron function works for report
        """
        return type(self)(kron(self.matrix, matrix.matrix))


    def __str__(self):
        """Provides a string representation if object is cast as such

        """
        return str(np.array(self.matrix.toarray()))

    def __getitem__(self, index):
        """Provides indexing of the object

        """
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

        if(isinstance(multiplier,SparseMatrix)):#Check if scalar or matrix multiplication valid
            if(isinstance(multiplier,Vector)): #If is multiplied by vector-like object will be 
                                               #(abstractly) a vector itself hence must return new 
                                               #vector (else same type return)
                return type(multiplier)(self.matrix*multiplier.matrix) 
            else:
                return type(self)(self.matrix*multiplier.matrix)#SEE __add__ for type(self) bit
        else:
            return type(self)(self.matrix.multiply(multiplier))#SEE __add__ for type(self) bit


    def __eq__(self,matrix):
        """Provides an equality comparison for Matrix like objects

        """
        if(self.matrix.nnz == matrix.matrix.nnz and \
           self.matrix.get_shape() == matrix.matrix.get_shape()): #Check number of items/shape is
                                                                #the same as a quick initial check
            if(type(self.matrix) != type(matrix.matrix)):
                try:
                    matrix = type(self)(matrix.matrix) #Allow for valid equivalence checking by 
                                                       #casting the matrix object as another, if
                                                       #cannot create other object with this ones
                                                       #matrix then cannot say they are the same
                                                       
                except:
                    return False # Can assume that if the data cannot be cast then not equal
            
            # Can compare components of the sparse array to check equal (also round data to 
            # account for calculation errors in the worst case)
       
            return all(np.around(self.matrix.data,4)==np.around(matrix.matrix.data,4))\
                   and all(self.matrix.indices == matrix.matrix.indices)\
                   and all(self.matrix.indptr  == matrix.matrix.indptr)

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
    """Provides representation of a square sparse matrix
    
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
        if(isinstance(matrix,SparseMatrix)):
            matrix = matrix.matrix # Can only construct from component matrix not from gates itself


        assert np.shape(matrix)[1] == np.shape(matrix)[0] #Ensure NxN matrix (i.e. square)
        
        
        super().__init__(matrix) #Call parent operator

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

        """
        shape = np.shape(matrix) #Set shape attribute
         
        if(len(shape) == 1):#Set a simple attribute to describe the `type' of vector
            self.type = "row"
        
        elif(shape[1]==1): #Ensure is either column or row vector 
            self.type = "col"
        else:
            raise ValueError("Cannot construct a vector with shape ",shape)
        
        self.matrix = csc_matrix(matrix)#We use this for vector as it allows for simpler working 
                                        #with measure later and also potentially more efficient 
                                        #vector workings
        self.matrix.eliminate_zeros()
        self.dimension = np.shape(matrix)
    
    def __getitem__(self,index): 
        if self.type == "row":#Access is different depending on vector type
            return self.matrix.A[0][index] #matrix stores as [[1,2,3,4]] so we do [0] first
        else:
            return self.matrix.A[index][0] #matrix stored as [[1],[2],[5]] so we do [0] second
   
    def __str__(self):
        #Overides the base class as we do not want to print row vectors as [[1,2]] but just [1,2]
        #Unless we do in which case remove
        if(self.type == "col"):
            return super().__str__()
            #return str(np.array(self.matrix.toarray()))
        else:
            return str(np.array(self.matrix.toarray()[0]))

