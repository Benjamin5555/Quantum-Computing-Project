"""Provides basic tests for the circuit_model implementation specifically the `bodged' kronker 
product

.. todo:: 
    * Requires improved documentation
    * Requires more rigorus testing

Author(s): 
 * Benjamin Carpenter(s1731178@ed.ac.uk)
"""


import unittest
from circuit_model_library import circuit_model 
import numpy as np
from scipy.sparse import csr_matrix




class TestGateKron(unittest.TestCase):


    #Hadamard Gate
    H = circuit_model.Gate(2**(-1/2) * np.array([[1,  1],\
                                                 [1, -1]]),"H")  
    
    #Pauli X Gate                                               
    X = circuit_model.Gate([[0, 1],\
                            [1, 0]],"X")                        
    
    #Identity Gate                                      
    I = circuit_model.Gate([[1, 0],\
                            [0, 1]],"I")
    
    
    ###Controlled not 
    #Controlled not for a 2 qubit system
    N = circuit_model.Gate([[1,0,0,0],\
                            [0,1,0,0],\
                            [0,0,0,1],\
                            [0,0,1,0]])
    
    
    #Pauli Z
    z = circuit_model.Gate(([[1,  0],\
                             [0, -1]]))
    
    #Controlled Pauli Z 
    Z = circuit_model.Gate(([[1,0,0,0],\
                             [0,1,0,0],\
                             [0,0,1,0],\
                             [0,0,0,-1]]))
    
    #Toffili Gate
    T = circuit_model.Gate([[1,0,0,0,0,0,0,0],\
                            [0,1,0,0,0,0,0,0],\
                            [0,0,1,0,0,0,0,0],\
                            [0,0,0,1,0,0,0,0],\
                            [0,0,0,0,1,0,0,0],\
                            [0,0,0,0,0,1,0,0],\
                            [0,0,0,0,0,0,0,1],\
                            [0,0,0,0,0,0,1,0]])
    
    c = circuit_model.Gate([[1,0],\
                            [0,1]],"c")

    gates_dictionary ={"I":I,\
                       "H":H,\
                       "X":X,\
                       "z":z,\
                       "Z":Z,\
                       "N":N,\
                       "T":T,\
                       "c":c}



    #Control top, controlled not gate
    
    
    test_circuit_string_list = ["HXIH","IXHI"]
    
    
    
    def test_kron_control(self):
        assert self.c.tensor_product(self.z) == self.Z


    def test_kron_double_control(self):
       a = self.c.tensor_product(self.X) 
       assert a== self.N
       assert (self.c.tensor_product(a).matrix.A==(self.T.matrix.A)).all() 

       
    def test_kron_over(self):
        a = (self.I.tensor_product(self.X)) 
        assert (self.c.tensor_product(a).matrix.A==circuit_model.Gate(\
                                               [[1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],\
                                                [0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ],\
                                                [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 ],\
                                                [0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 ],\
                                                [0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 ],\
                                                [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 ],\
                                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 ],\
                                                [0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 ]]).matrix.A).all()
    
    def test_circuit_over_with_bodge(self):
        test_string = ["c",\
                       "I",\
                       "X"]

        prod_circ = circuit_model.QuantumCircuit(test_string,self.gates_dictionary)
        expected_matrix = csr_matrix([[1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],\
                                      [0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ],\
                                      [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 ],\
                                      [0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 ],\
                                      [0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 ],\
                                      [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 ],\
                                      [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 ],\
                                      [0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 ]])

        expc_circ = circuit_model.QuantumCircuit(expected_matrix)

        assert (expc_circ.matrix.A == prod_circ.matrix.A).all()

    
    def test_circuit_mult_contol_with_bodge(self):#Testing creating a Toffoli gate from components
        test_string = ["c",\
                       "c",\
                       "X"]

        prod_circ = circuit_model.QuantumCircuit(test_string,self.gates_dictionary)
        
        expc_circ = circuit_model.QuantumCircuit(["ITI","III","III"],self.gates_dictionary)

        assert (expc_circ.matrix.A == prod_circ.matrix.A).all()

    
