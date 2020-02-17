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
    def test_quantum_circuit_creation(self):
        """

        """
        H = circuit_model.Gate("H",2**(-1/2) * np.array([[1,  1],\
                                                         [1, -1]]))

        N = circuit_model.Gate("N",[[0, 1],\
                                    [1, 0]])

        I = circuit_model.Gate("I",[[1, 0],\
                                    [0, 1]])



        test_circuit_string_list = ["HXIH","IXHI"]
        test_gates_dictionary = {"I":I,\
                                 "H":H,\
                                 "n":"|controlled not head|",\
                                 "c":"|controlled not control|",\
                                 "X":N}

        circuit1 = circuit_model.QuantumCircuit(test_circuit_string_list, test_gates_dictionary)
        print(circuit1.circuit.A)
        print("----------------------")
        test_circuit_not_string = ["XXI","XIX"]
        circuit2 = circuit_model.QuantumCircuit(test_circuit_not_string, test_gates_dictionary)
        #Should return identity matrix
        print(circuit2.circuit.A) 
        """
        assert circuit2.circuit.A == Gate(np.array([[1,0,0,0],\
                                                    [0,1,0,0],\
                                                    [0,0,1,0],\
                                                    [0,0,0,1]]))
                                                    """
        print("----------------------")
        circuit_string3 = ["HI","IX"]
        circuit3 =circuit_model.QuantumCircuit(circuit_string3  , test_gates_dictionary)
        print(circuit3.circuit.A)
