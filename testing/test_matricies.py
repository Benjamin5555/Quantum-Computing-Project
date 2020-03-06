"""Provides basic tests for the matrices implementation



.. todo:: Check vector conjugate & transpose

Author(s): 
 * Benjamin Carpenter(s1731178@ed.ac.uk)
"""

import unittest
from circuit_model_library import matrices 
import numpy as np

class TestSquareMatrixImp(unittest.TestCase):
    """

    """
    def create_SquareMatrix(self):
        mc = np.array([[2, 3,  0, 4  ],\
                       [3, 1,  9, 0  ],\
                       [5, 0,  2, 1  ],\
                       [0, 2, 11, 90 ]])

        mc2 =np.array([[102,0,0  ,0 ],\
                       [0  ,1,0  ,0 ],\
                       [0  ,0,221,0 ],\
                       [0  ,0,0  ,90]])

        testSqMatrixA = matrices.SquareMatrix(mc)    
        testSqMatrixB = matrices.SquareMatrix(mc2)
        
        return testSqMatrixA,testSqMatrixB,mc,mc2

    def test_create_SquareMatrix(self):
        self.create_SquareMatrix()


    def test_SquareMatrix_access(self):
        testSqMatrixA, testSqMatrixB, *_ = self.create_SquareMatrix()
        assert testSqMatrixA[1][0]==3
        assert testSqMatrixB[3][2]==0


    def test_SquareMatrix_basic_operations(self):
        testSqMatrixA, testSqMatrixB, mc, mc2 = self.create_SquareMatrix()
        a = testSqMatrixA + testSqMatrixB
        b = matrices.SquareMatrix(mc+mc2)
        assert (testSqMatrixA + testSqMatrixB) == matrices.SquareMatrix(mc+mc2)
        assert (testSqMatrixA - testSqMatrixB) == matrices.SquareMatrix(mc-mc2)
       
        tp = (testSqMatrixA * testSqMatrixB)
        tx = matrices.SquareMatrix(np.matmul(mc,mc2))
        
        assert (testSqMatrixA * testSqMatrixB) == matrices.SquareMatrix(np.matmul(mc,mc2))
        assert isinstance((testSqMatrixA + testSqMatrixB),matrices.SquareMatrix) 

    def test_SquareMatrix_conjugate_transpose(self):

        hermatian_matrix = matrices.SquareMatrix([[   2, 2+1j,  4],\
                                                  [2-1j,   3, 1j],\
                                                  [   4,  -1j,  1]])

        assert hermatian_matrix == hermatian_matrix.conjugate_transpose()
    
    def create_Vectors(self):
        rv = np.array([2, 0,  0, 4  ])
                       
        cv = np.array([[34],\
                        [2],\
                        [4],\
                        [5]])



        cv2 =np.array([[102],\
                         [0],\
                         [4],\
                         [0]])
        
        testCVectorA = matrices.Vector(cv) #Testing creation    
        testCVectorB = matrices.Vector(cv2)
        testRVector =  matrices.Vector(rv)
        return testCVectorA,testCVectorB,testRVector,rv,cv,cv2
    
    def test_vector_elem_slice(self):
        a = [1,2,3,4,0,0,6,4]
        testVector = matrices.Vector(a)
        for i in range(len(a)):
            assert a[i] == testVector[i]

    def test_vector_shape(self):
        a = [1,2,3,4,0,0,6,4]
        testVector = matrices.Vector(a)
        np.shape(testVector) == (np.shape(a[0]),1)

    def test_create_vectors(self):
        self.create_Vectors()

    def test_Vector_access_operations(self):
        """Testing accessing elements within a vector
        
        """
         
        testCVectorA, testCVectorB, testRVector,*_ = self.create_Vectors()

        assert testCVectorA[1] == 2 #(Bad) testing acccesses 

        assert testRVector[2]==0 #Ditto

    def test_Vector_basic_operations(self):
        testCVectorA, testCVectorB, testRVector,rv, cv,cv2 = self.create_Vectors()
        testSqMatrixA, testSqMatrixB, mc, mc2 = self.create_SquareMatrix()

        a = testCVectorA + testCVectorB
        b = matrices.Vector(cv+cv2)
        assert (testCVectorA + testCVectorB) == matrices.Vector(cv+cv2)
        assert (testCVectorA - testCVectorB) == matrices.Vector(cv-cv2)

        assert (testSqMatrixA  * testCVectorA) == matrices.SparseMatrix(np.matmul(mc,cv))



    def tensor_test_def(self):
       a = matrices.SparseMatrix([[1, 0, -1, 0], [-1, 1, 1, 0], [2, 4, -2, 1], [0, 0, 0, 1]])
       b = matrices.SparseMatrix([[1, 0], [1, 1]])
       c = matrices.SparseMatrix([[5, 1,   2, 3], [12, 31, 94, 21], [134, 34, 5, 2], [3, 1, 1, 0]]) 
       d = matrices.SparseMatrix([[-1,1],[-1,0]])

       #tenorproduct of b,d
       o1 =  [[-1, 1,  0, 0],\
              [-1, 0,  0, 0],\
              [-1, 1, -1, 1],\
              [-1, 0, -1, 0]]
       o1 = matrices.SparseMatrix(o1)

       #o2 = tensorproduct of a,b
       o2 =   [[1, 0, 0, 0, -1, 0, 0, 0], [1, 1, 0, 0, -1, -1, 0, 0], [-1, 0, 1, 0, \
               1, 0, 0, 0], [-1, -1, 1, 1, 1, 1, 0, 0], [2, 0, 4, 0, -2, 0, 1,      \
               0], [2, 2, 4, 4, -2, -2, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0,  \
               0, 0, 0, 1, 1]]
              
       o2 = matrices.SparseMatrix(o2)

       #tensor product of b,a
       o3 =[[1, 0, -1, 0, 0, 0, 0, 0], [-1, 1, 1, 0, 0, 0, 0, 0], [2, 4, -2, 1, \
            0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, 0, -1, 0, 1, 0, -1,      \
            0], [-1, 1, 1, 0, -1, 1, 1, 0], [2, 4, -2, 1, 2, 4, -2, 1], [0, 0,  \
            0, 1, 0, 0, 0, 1]]       
       
       
       o3 = matrices.SparseMatrix(o3)

       #Tensor product of a,c
       o4 = [[5, 1, 2, 3, 0, 0, 0, 0, -5, -1, -2, -3, 0, 0, 0, 0], [12, 31, 94, \
          21, 0, 0, 0, 0, -12, -31, -94, -21, 0, 0, 0, 0], [134, 34, 5, 2, 0,    \
          0, 0, 0, -134, -34, -5, -2, 0, 0, 0, 0], [3, 1, 1, 0, 0, 0, 0,         \
          0, -3, -1, -1, 0, 0, 0, 0, 0], [-5, -1, -2, -3, 5, 1, 2, 3, 5, 1, 2,   \
           3, 0, 0, 0, 0], [-12, -31, -94, -21, 12, 31, 94, 21, 12, 31, 94,      \
          21, 0, 0, 0, 0], [-134, -34, -5, -2, 134, 34, 5, 2, 134, 34, 5, 2,     \
          0, 0, 0, 0], [-3, -1, -1, 0, 3, 1, 1, 0, 3, 1, 1, 0, 0, 0, 0,          \
          0], [10, 2, 4, 6, 20, 4, 8, 12, -10, -2, -4, -6, 5, 1, 2, 3], [24,     \
          62, 188, 42, 48, 124, 376, 84, -24, -62, -188, -42, 12, 31, 94,        \
          21], [268, 68, 10, 4, 536, 136, 20, 8, -268, -68, -10, -4, 134, 34,    \
          5, 2], [6, 2, 2, 0, 12, 4, 4, 0, -6, -2, -2, 0, 3, 1, 1, 0], [0, 0,    \
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 2, 3], [0, 0, 0, 0, 0, 0, 0, 0,    \
          0, 0, 0, 0, 12, 31, 94, 21], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      \
          134, 34, 5, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 0]]
       
       o4 = matrices.SparseMatrix(o4)
         
       return a,b,c,d,o1,o2,o3,o4


    def test_tensor_product(self):
       a,b,c,d,o1,o2,o3,o4 = self.tensor_test_def()

       # b tensor d 
       assert b.tensor_product(d) == o1 
       # a ten   b

       assert a.tensor_product(b) == o2
       # b teso  a 
       assert b.tensor_product(a) == o3 
       # a tens c
       assert a.tensor_product(c) == o4 

    if __name__ == 'main':

        unittest.main()
