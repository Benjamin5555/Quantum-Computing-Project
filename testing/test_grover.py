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
import matplotlib.pyplot as plt



class TestGrover(unittest.TestCase):
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

    def test_grover_5_qubit(self):
        reg_5_qubit = circuit_model.QuantumRegister([1],(2**5,1))
        grov_5_qubit_string = ["HXcXHXcXHII",\
                               "HIcIHXcXHII",\
                               "HXcXHXcXHII",\
                               "HXcXHXzXHII",\
                               "HIXIIIIIIHX"]
        
                          
        #print(grov_5_qubit_string)
        Grov_5_qbit = circuit_model.QuantumCircuit(grov_5_qubit_string,self.gates_dictionary) 
        a = Grov_5_qbit.apply(reg_5_qubit)
        plt.plot(a.measure()[0],a.measure()[1])
        plt.show()


