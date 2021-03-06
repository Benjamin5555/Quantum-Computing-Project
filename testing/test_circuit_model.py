"""Provides basic tests for the circuit_model implementation

.. todo:: 
    * Requires improved documentation
    * Requires more rigorus testing

Author(s): 
 * Benjamin Carpenter(s1731178@ed.ac.uk)
 * Gregor Rowley (s1705159@ed.ac.uk)
"""

import unittest
from circuit_model_library import circuit_model 
import numpy as np
from scipy.sparse import csr_matrix
import inspect




class TestCircuitModel(unittest.TestCase):
    """
    Tests for circuit_model.
    """
    # Quantum logic gates, represented by matrices:

    # Hadamard gate:
    H = circuit_model.Gate(2**(-1/2) * np.array([[1,  1],\
                                                 [1, -1]]))

    # Negation/Pauli-X gate:
    N = circuit_model.Gate([[0, 1],\
                            [1, 0]])                        

    # Identity gate:                                               
    I = circuit_model.Gate([[1, 0],\
                            [0, 1]])
    
    # Pauli-Z gate:
    z = circuit_model.Gate([[1 ,0],\
                            [0 ,-1]])

    c = circuit_model.Gate([[1,0],\
                            [0,1]],"c")

    # Controlled Z (CZ) gate:
    Z = circuit_model.Gate([[1, 0, 0, 0],\
                            [0, 1, 0, 0],\
                            [0, 0, 1, 0],\
                            [0, 0, 0,-1]])


    
    # List of string's representing gate combinations:
    test_circuit_string_list = ["HXIH","IXHI"]
    # Dictionary of gate ID's:
    test_gates_dictionary = {"I":I,\
                             "H":H,\
                             "X":N,\
                             "z":z,\
                             "Z":Z,\
                             "c":c}
    
    
    def __setup_test(self):
        """
        Sets up the quantum registers and gate combinations strings
        required for testing.
        """
        test_register_01 = circuit_model.QuantumRegister([1],shape = (4,1))
        test_circuit_not_not_string = ["XXI","XIX"]
        circuit2 = circuit_model.QuantumCircuit(test_circuit_not_not_string, self.test_gates_dictionary)
        return test_register_01,circuit2


    def test_run_circuit(self):
        """
        Tests the register instantiated in the __setup_test method,
        applied to the NOT NOT circuit combination, also defined above,
        to ensure that it returns the original register, since the double 
        negative of the two NOT gates should produce no result.
        """
        register, not_not_circuit = self.__setup_test()
        assert not_not_circuit*register == register
        

    def test_quantum_register_creation(self):
        """
        Creates the test register required for further tests below.
        """
        test_register = circuit_model.QuantumRegister([0,1,4,9],shape = (12,1))



    def test_Hadamard_run(self):
        """
            Tests that the simulated  'circuit' process works for most basic gate i.e. Hadamard
        """
        
        """
            Apply to Hadamard gate to single qubit
        """
        test_register_00 = circuit_model.QuantumRegister([0],shape=(4,1))
        circuit_single_Hadamard = circuit_model.QuantumCircuit(["II","HI"], self.test_gates_dictionary)

        out_register = circuit_single_Hadamard.apply(test_register_00)

        p_calc = np.around(out_register.measure()[1],4)

        p_exp = [0.5,0.5,0,0] 
        # Expect equal prob of first bit being one as we applied Hadamard to the first bit only
        assert (p_calc == [0.5,0.5]).all()
        
        assert (out_register.measure()[0] == [0,1]).all()

         
        #I.e. apply Hadamard to single qubit of |00> we expect equal probability of |0> and |1> and 
        # other bit constant |0> state 

        """
            Apply a Hadamard to both qubits 
        """
        circuit_string = ["HI","IH"]

        test_register_00 = circuit_model.QuantumRegister([0],shape = (4,1)) #Test with |00> State
        circuit_Hadamard = circuit_model.QuantumCircuit(circuit_string, self.test_gates_dictionary)
        out_register =circuit_Hadamard.apply(test_register_00)
        

        assert(all(round(p,4) == 0.25 for p in out_register.measure()[1]))
        #I.e. apply hadamard to |00> we expect equal probability of every state
       

    def basic_circuit_creation_definitions(self):
        """
        Creates the Quantum Registers and Quantum Circuits which are required,
        for testing circuit creation.
        """
        qu_reg_00 = circuit_model.QuantumRegister([0],(4,1))
        qu_reg_10 = circuit_model.QuantumRegister([2],(4,1))

        expected_1 = circuit_model.QuantumCircuit(\
                csr_matrix(\
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]))

        gen_1 = circuit_model.QuantumCircuit(["IcI","HXH"],self.test_gates_dictionary)

        expected_2 = circuit_model.QuantumCircuit(\
                csr_matrix(\
                [[1/2, 1/2, 1/2, -(1/2)], [1/2, 1/2, -(1/2), 1/2], [1/2, -(1/2), 1/2,\
                1/2], [-(1/2), 1/2, 1/2, 1/2]]))
        gen_2 = circuit_model.QuantumCircuit(["HcH","IXI"],self.test_gates_dictionary)



        return qu_reg_00,qu_reg_10,expected_1,gen_1,expected_2,gen_2

    def test_basic_circuit_creation(self):
        """
        Tests the creation of basic quantum circuits, checking that they match
        the expected results.
        """
        qu_reg_00,qu_reg_10,expected_1,gen_1,*_= self.basic_circuit_creation_definitions()
        assert (expected_1 == gen_1)


    def test_intermediate_circuit_Application(self):
        """
        Tests the application of quantum circuits ensuring the states and 
        lengths match expected results.
        """
        qu_reg_00,qu_reg_10,expected_1,gen_1,expected_2,gen_2 =\
                self.basic_circuit_creation_definitions()
        #EFFECTIVELY AN IDENTITY MATRIX
        assert (gen_1.apply(qu_reg_00).measure()[0]\
                == expected_1.apply(qu_reg_00).measure()[0]).all()

        assert (np.around(gen_1.apply(qu_reg_00).measure()[1],4)\
                == np.around(expected_1.apply(qu_reg_00).measure()[1],4)).all()
        
        assert (gen_1.apply(qu_reg_00).measure()[0]\
                == [0]).all() 

        assert (np.around(gen_1.apply(qu_reg_00).measure()[1],4)\
                == [1]).all()

        assert (gen_1.apply(qu_reg_10).measure()[0]\
                == [2]).all() 

        assert (np.around(gen_1.apply(qu_reg_10).measure()[1],4)\
                == [1]).all()
        #NOT EFFECTIVELY AN IDENTITY MATRIX
        assert (gen_2.apply(qu_reg_00).measure()[0]\
                == expected_2.apply(qu_reg_00).measure()[0]).all()

        assert (np.around(gen_2.apply(qu_reg_00).measure()[1],4)\
                == np.around(expected_2.apply(qu_reg_00).measure()[1],4)).all()
        
        assert (gen_2.apply(qu_reg_00).measure()[0]\
                == [0,1,2,3]).all() 

        assert (np.around(gen_2.apply(qu_reg_00).measure()[1],4)\
                == [0.25,0.25,0.25,0.25]).all()

       

    def test_grovers_c_00(self):
        """
        Tests Grover's algorithm with register 00, ensuring the quantum register 
        matches expected results, after Grover's algorithm has been applied.
        """
        test_grovers_00 = ["HXcXHzcH",\
                           "HXzXHzzH"]

        circuit_Grover_00 = circuit_model.QuantumCircuit(test_grovers_00,\
                                                          self.test_gates_dictionary)
        assert (np.around(circuit_Grover_00.matrix.A,4) == np.array([[1,  0,  0,  0],\
                                                     [0,  0, -1,  0],\
                                                     [0, -1,  0,  0],\
                                                     [0,  0,  0, -1]])).all
        qu_reg_00 = circuit_model.QuantumRegister([0],(4,1))
        out_register = circuit_Grover_00.apply(qu_reg_00)
        assert (out_register.measure()[0] == [0]).all
        assert (np.around(out_register.measure()[1],4) == [1]).all



    def test_grovers_c_01(self):
        """
        Tests Grover's algorithm with register 01, ensuring the quantum register 
        matches expected results, after Grover's algorithm has been applied.
        """
        test_grovers_01 = ["HXcXHzcH",\
                           "HIzIHzzH"]

        circuit_Grover_01 = circuit_model.QuantumCircuit(test_grovers_01,\
                                                         self.test_gates_dictionary)
        
        assert (np.around(circuit_Grover_01.matrix.A,4) == np.array([[0,  0, -1, 0],\
                                                      [1,  0,  0, 0],\
                                                      [0,  0,  0, 1],\
                                                      [0,  1,  0, 0]])).all()
        
        qu_reg_00 = circuit_model.QuantumRegister([0],(4,1))
        out_register = circuit_Grover_01.apply(qu_reg_00)
        assert out_register.measure()[0] == [1]
        assert np.around(out_register.measure()[1],4) == [1]

    def test_grovers_c_10(self):
        """
        Tests Grover's algorithm with register 10, ensuring the quantum 
        register matches expected results, after Grover's algorithm has been 
        applied.
        """
        test_grovers_10 = ["HIZIHzZH",\
                           "HXZXHzZH"]

        circuit_Grover_10 = circuit_model.QuantumCircuit(test_grovers_10,\
                                                          self.test_gates_dictionary)


        assert (np.around(circuit_Grover_10.matrix.A,4) == np.array(\
                                                    [[ 0, -1,  0,  0],\
                                                     [ 0,  0,  0,  1],\
                                                     [ 1,  0,  0,  0],\
                                                     [ 0,  0,  1,  0]])).all()
        
        qu_reg_00 = circuit_model.QuantumRegister([0],(4,1))
        out_register = circuit_Grover_10.apply(qu_reg_00)
        assert out_register.measure()[0] == [2]
        assert np.around(out_register.measure()[1],4) == [1]

    def test_grovers_c_11(self):            
        """
        Tests Grover's algorithm with register 11, ensuring the quantum 
        register matches expected results, after Grover's algorithm has been 
        applied.
        """
        test_grovers_11 = ["HIZIHzZH",\
                           "HIZIHzZH"]

        circuit_Grover_11 = circuit_model.QuantumCircuit(test_grovers_11,\
                                                          self.test_gates_dictionary)

        assert (np.around(circuit_Grover_11.matrix.A,4) == np.array(\
                                                    [[ 0,  0,  0, - 1],\
                                                     [ 0,  1,  0,   0],\
                                                     [ 0,  0,  1,   0],\
                                                     [ 1,  0,  0,   0]])).all()
        qu_reg_00 = circuit_model.QuantumRegister([0],(4,1))
        out_register = circuit_Grover_11.apply(qu_reg_00)
        assert out_register.measure()[0] == [3]
        assert np.around(out_register.measure()[1],4) == [1]

