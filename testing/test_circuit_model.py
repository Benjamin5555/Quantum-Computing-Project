"""Provides basic tests for the circuit_model implementation

.. todo:: 

Author(s): 
 * Benjamin Carpenter(s1731178@ed.ac.uk)
"""

import unittest
from circuit_model_library import circuit_model 
import numpy as np
import inspect

class TestCircuitModel(unittest.TestCase):
    """

    """

    H = circuit_model.Gate(2**(-1/2) * np.array([[1,  1],\
                                                 [1, -1]]))
                                                            
    N = circuit_model.Gate([[0, 1],\
                            [1, 0]])                        
                                                            
    I = circuit_model.Gate([[1, 0],\
                            [0, 1]])
    
    
    
    test_circuit_string_list = ["HXIH","IXHI"]
    test_gates_dictionary = {"I":I,\
                             "H":H,\
                             "n":"|controlled not head|",\
                             "c":"|controlled not control|",\
                             "X":N}
    
    
    def __setup_test(self):
        test_register = circuit_model.QuantumRegister([1],shape = (4,1))
        test_circuit_not_not_string = ["XXI","XIX"]
        circuit2 = circuit_model.QuantumCircuit(test_circuit_not_not_string, self.test_gates_dictionary)
        return test_register,circuit2


    def test_run_circuit(self):
        register, not_not_circuit = self.__setup_test()
        print("-----------------------")
        print(type(not_not_circuit*register))
        print(not_not_circuit)
        print(register)
        print("-------____")
        print(not_not_circuit*register)
        assert not_not_circuit*register == register
        

    def test_quantum_register_creation(self):
        """

        """
        test_register = circuit_model.QuantumRegister([0,1,4,9],shape = (12,1))
        print("------------------")
        print(test_register)



    def test_quantum_circuit_creation(self):
        """

        """
        circuit1 = circuit_model.QuantumCircuit(self.test_circuit_string_list, self.test_gates_dictionary)
        print(circuit1)
        print("----------------------")
        test_circuit_not_string = ["XXI","XIX"]
        circuit2 = circuit_model.QuantumCircuit(test_circuit_not_string, self.test_gates_dictionary)
        #Should return identity matrix
        print(circuit2) 
        """
        assert circuit2.circuit.A == Gate(np.array([[1,0,0,0],\
                                                    [0,1,0,0],\
                                                    [0,0,1,0],\
                                                    [0,0,0,1]]))
                                                    """
        print("----------------------")
        circuit_string3 = ["HI","IX"]
        circuit3 =circuit_model.QuantumCircuit(circuit_string3  , self.test_gates_dictionary)
        print(circuit3)
