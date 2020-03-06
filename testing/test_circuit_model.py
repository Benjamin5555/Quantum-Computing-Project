"""Provides basic tests for the circuit_model implementation

.. todo:: 

Author(s): 
 * Benjamin Carpenter(s1731178@ed.ac.uk)
"""

import unittest
from circuit_model_library import circuit_model 
import numpy as np
from scipy.sparse import csr_matrix
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
    
    z = circuit_model.Gate([[1 ,0],\
                            [0 ,-1]])

    Z = circuit_model.Gate([[1, 0, 0, 0],\
                            [0, 1, 0, 0],\
                            [0, 0, 1, 0],\
                            [0, 0, 0,-1]])


    #Control top, controlled not gate
    c = circuit_model.Gate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    
    
    test_circuit_string_list = ["HXIH","IXHI"]
    test_gates_dictionary = {"I":I,\
                             "H":H,\
                             "X":N,\
                             "z":z,\
                             "Z":Z,\
                             "c":c}
    
    
    def __setup_test(self):
        test_register_01 = circuit_model.QuantumRegister([1],shape = (4,1))
        test_circuit_not_not_string = ["XXI","XIX"]
        circuit2 = circuit_model.QuantumCircuit(test_circuit_not_not_string, self.test_gates_dictionary)
        return test_register_01,circuit2


    def test_run_circuit(self):
        register, not_not_circuit = self.__setup_test()
        assert not_not_circuit*register == register
        

    def test_quantum_register_creation(self):
        """

        """
        test_register = circuit_model.QuantumRegister([0,1,4,9],shape = (12,1))



    def test_quantum_circuit_creation(self):
        """

        """
        
        
        circuit1 = circuit_model.QuantumCircuit(self.test_circuit_string_list, self.test_gates_dictionary)
        test_circuit_not_string = ["XXI","XIX"]
        
        
        
        circuit2 = circuit_model.QuantumCircuit(test_circuit_not_string, self.test_gates_dictionary)
        #Should return identity matrix

        assert (circuit2.matrix.A == (np.array([[1,0,0,0],\
                                                [0,1,0,0],\
                                                [0,0,1,0],\
                                                [0,0,0,1]]))).all()
        circuit_string3 = ["HI","IX"]
        circuit3 =circuit_model.QuantumCircuit(circuit_string3  , self.test_gates_dictionary)
        circuit3Expected = circuit_model.QuantumCircuit([[0, 1/np.sqrt(2), 0, 1/np.sqrt(2)   ],\
                                                       [1/np.sqrt(2), 0, 1/np.sqrt(2), 0   ],\
                                                       [0, 1/np.sqrt(2), 0, -(1/np.sqrt(2))],\
                                                       [1/np.sqrt(2), 0, -(1/np.sqrt(2)), 0]])

        assert circuit3== circuit3Expected

        
    def test_large_string(self):
        circuit_string4 = test_circuit_not_string = ["XXI","XIX","IXX","IXX"]
        circuit4 =circuit_model.QuantumCircuit(circuit_string4  , self.test_gates_dictionary)
        test_register_00 = circuit_model.QuantumRegister([0],shape = (16,1))
        assert test_register_00 == circuit4.apply(test_register_00) #Will be equal due to XX = I 



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
        #print(out_register.measure()[0])

        #print(out_register.measure()[1])
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
        qu_reg_00 = circuit_model.QuantumRegister([0],(4,1))
        qu_reg_10 = circuit_model.QuantumRegister([2],(4,1))

        expected_1 = circuit_model.QuantumCircuit(\
                csr_matrix(\
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]))

        gen_1 = circuit_model.QuantumCircuit(["IcI","HcH"],self.test_gates_dictionary)

        expected_2 = circuit_model.QuantumCircuit(\
                csr_matrix(\
                [[1/2, 1/2, 1/2, -(1/2)], [1/2, 1/2, -(1/2), 1/2], [1/2, -(1/2), 1/2,\
                1/2], [-(1/2), 1/2, 1/2, 1/2]]))
        gen_2 = circuit_model.QuantumCircuit(["HcH","IcI"],self.test_gates_dictionary)



        return qu_reg_00,qu_reg_10,expected_1,gen_1,expected_2,gen_2

    def test_basic_circuit_creation(self):
        qu_reg_00,qu_reg_10,expected_1,gen_1,*_= self.basic_circuit_creation_definitions()
        assert (expected_1 == gen_1)


    def test_intermediate_circuit_Application(self):

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
        #print("TEST GROVERS 00")
        
        test_grovers_00 = ["HXZXHzZH",\
                           "HXZXHzZH"]

        circuit_Grover_00 = circuit_model.QuantumCircuit(test_grovers_00,\
                                                          self.test_gates_dictionary)

        #print(circuit_Grover_00.matrix.A)
        #print("PRE ASSERT")
        assert (circuit_Grover_00.matrix == np.array([[1,  0,  0,  0],\
                                                     [0,  0, -1,  0],\
                                                     [0, -1,  0,  0],\
                                                     [0,  0,  0, -1]])).all

    def test_grovers_c_01(self):
        print("Relevant Test")
        test_grovers_01 = ["HIZIHzZH",\
                           "HXZXHzZH"]

        circuit_Grover_01 = circuit_model.QuantumCircuit(test_grovers_01,\
                                                          self.test_gates_dictionary)


        print("PRE ASSERT")
        assert (circuit_Grover_01.matrix == np.array([[ 0,  0,  1,  0],\
                                                     [-1,  0,  0,  0],\
                                                     [ 0,  0,  0,  1],\
                                                     [ 0,  1,  0,  0]])).all()

    def test_grovers_c_10(self):
        test_grovers_10 = ["HXZXHzZH",\
                           "HIZIHzZH"]

        circuit_Grover_10 = circuit_model.QuantumCircuit(test_grovers_10,\
                                                          self.test_gates_dictionary)


        assert (circuit_Grover_10.matrix == np.array([[ 0,  1,  0,  0],\
                                                     [ 0,  0,  0,  1],\
                                                     [-1,  0,  0,  0],\
                                                     [ 0,  0,  1,  0]])).all()

    def test_grovers_c_11(self):            
        #print("TEST GROVER |11>")
        
        test_grovers_11 = ["HIZIHzZH",\
                           "HIZIHzZH"]

        circuit_Grover_11 = circuit_model.QuantumCircuit(test_grovers_11,\
                                                          self.test_gates_dictionary)

        #print(circuit_Grover_11.matrix.A)
        #print("PRE ASSERT")
        assert (circuit_Grover_11.matrix == np.array([[ 0,  0,  0,  1],\
                                                     [ 0,  1,  0,  0],\
                                                     [ 0,  0,  1,  0],\
                                                     [-1,  0,  0,  0]])).all()
