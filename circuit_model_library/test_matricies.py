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
        assert (testSqMatrixA * testSqMatrixB) == matrices.SquareMatrix(mc*mc2)
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

        a = testCVectorA + testCVectorB
        b = matrices.Vector(cv+cv2)
       
        assert (testCVectorA + testCVectorB) == matrices.Vector(cv+cv2)
        assert (testCVectorA - testCVectorB) == matrices.Vector(cv-cv2)
        assert (testCVectorA * testCVectorB) == matrices.Vector(cv*cv2)

    if __name__ == 'main':

        unittest.main()
