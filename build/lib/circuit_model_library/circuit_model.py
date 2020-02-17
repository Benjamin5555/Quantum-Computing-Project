"""This module details the main and specific functionality of the circuit model including 
a functionality that will 'run' a circuit
.. todo::
    * Implementation of gate given a decision on exactly how it will be called/used
Author(s): 
 * Benjamin Carpenter(s1731178@ed.ac.uk)
"""

from circuit_model_library import matrices
import scipy.sparse
import numpy as np

class Gate(matrices.SquareMatrix):
    """A class to implement gates, will share a large amount of functionality with a square matrix
    but will require have a more specific criteria for what form a matrix can take as well as
    likely requirng additional functionality
    
    Attributes:
        gate_id: A unique identifier of the type of gate that the instance is
        matrix: A matrix representing the operation of the gate
    .. todo::
        *Add verification so actual constraints on what a gate can be are observed (i.e. unitary)
    """
    def __init__(self, gate_id, matrix):
        """Creates a gate object
        Args:
        Raises:
            ValueError: On invalid form for a Quantum circuit gate
        """
        #assert Hermatian, other requirments for a gate here

        #Call the parent class constructor after verifying valid parameters for a gate  
        self.gate_id = gate_id
        super(self,matrix)

class QuantumRegister(matrices.Vector):
    """A system of multiple qubits, abstractly the tensor product of multiple qubits .
    Practically equivalent to the volatile memory of a CPU in a classical computer
    Attributes:
        register: The tensor product of quBits in the sytem under consideration
    
    """
    def __init__(self):
        """Initialise a quantum register with specific values
        Args:
            register_initial_state: Values related to the initial state of the regitster
        """
        NotImplemented 





class QuantumCircuit():
    """An object representing a quantum circuit, can be used to run circuits on ...
    
    Constructs circuit from a given set of gates and schematic using the fact that a series of gates
    is the dot product of the gates (in reverse order i.e. applying \\(\hat{A} -> \hat{B}\\) is 
    equivlent to applying the gate \\(\hat{C} := \hat{B}\cdot\hat{A} \\) and applying parallel gates 
    to the register is equivalent to the tensor product of these gates. By use of the Identity 
    matrix we can create a matrix representing the circuit
    """
    def __init__(self, circuit_string_list, gates_dictionary):
        """Creates a quantum circuit which a register can be run on
        
        Args:
            circuit_string_list: A list of strings representing each row of the quantum circuit 
            where each character in the string relates to (or part of) a  quantum gate
            gates_dictionary: A dictionary of gate objects relating to those used in the circuit 
            with id's matching that of those used in the circuit string 
        
        Raises:
            TypeError: If non list of strings argument given for the circit_string_list
        """
        
        #Reverse each row for dotting and applying in correct order later on 
        returnList = []
        for x in circuit_string_list:
            returnList.append(list(x[::-1])) # Reverses the string in place and convert to list 
                                             # e.g. "ABC" -> ['A','B','C']
        #Restructure the data set from [[a_1,a_2,...],[b_1,b_2,...],[c_1,c_2,...]] 
        #to [[a_1,b_1,c_1],[a_2,b_2,c_3].[a_3,b_3,c_3],...]
        circuit_string_list = self._reform_data(returnList)
        
        #Convert to actual gates rather than ID's
        circuit_matrix_comp = []
        
        #TESTING##############################
         
        print(returnList)
        print("----")
        print(np.array(circuit_string_list))
        print("----")
        ######################################
        
        
        for gate_column in circuit_string_list:# Go through each paralell gate in the list of 
                                               # paralell gates and replace each element with an 
                                               # matrix representing its gate, then 
                                               # tensor product together the column
            current_column = []


            # Go through the whole column and replace each gate_id with actual gate matrix
            print(gate_column)
            for gate_id in gate_column:
                
                current_column.append(gates_dictionary[gate_id]) #'Replace the gate_id with the 
                                                                 # actual gate
            print(current_column[0],current_column[1])
            
            #Get tensor product of each 'column'

            #Tensor product together the first two elelments of our 'column' so we have something to
            #work on with the for loop for the rest of the 'column'   
            column_product = current_column[0].tensor_product(current_column[1]) 

            #Apply tensor product to rest of the 'column' (though only if there are more elements)
            if len(current_column) > 2: 
                for gate in current_column[2:]: 
                    column_product = column_product.tensor_product(gate) 
            #Add the column (now a single matrix) back to a list so it can be dotted together at 
            #a later point with the rest of the circuit matrix
            circuit_matrix_comp.append(column_product)             
            #print(circuit_matrix_comp[-1])#Testing

        print(np.array(circuit_matrix_comp)) 
        #!!!!consider using fold left / right  !!!

        """circuit_matrix_interval_comp = []
        for gate_column in circuit_matrix_comp:
            for gate_column_row in gate_column:
        circuit_matrix_interval_comp.append(
        """
        #Dot each of these together 

        #Bam, a quCircuit that can be applied to a quRegister

    @staticmethod
    def _reform_data(oppDat):
        """Helper function that takes a list of lists and returns an arrays of a single variables 
        i.e. [[a1,b1,c1,d1],[a2,b2,c2,d2]]  -> [[a1,a2],[b1,b2],[c1,c2],[d1,d2]]
        """
        lenIndi = len(oppDat[0])
        rDat = []
        for i in range(len(oppDat[0])):#Better ways of doing but creates an empty 
                                       #array of data
            rDat.append([])
        oppDat = np.array(oppDat).flatten()#Create 1D long list from given 2D list
        for i in range(len(oppDat)): 
            rDat[i%lenIndi].append(oppDat[i]) #Append value to the correct 
                                              #position using mod
        return rDat  


class Qubit(matrices.Vector):
    """A vector like object that represents a quantum bit (qubit)
    Attributes:
        state: A 2 dimensional vector representing the state of the Qubit
    .. todo:: Weirdly this may not be a required class consider whether it is needed
    """
    def __init__(self, bit_positions):
        """Creates a qubit object.
        A quantum bit (qubit) is a system that can be observed in two unique states such as electron
        spin (can be up or down).
        A qubit is a vector in a Hilbert space (n-dimensional vector space) where \\(n\\) quibits
        will represent \\(2^n\\) dimensional Hilbert space.
        Each state is a basis of the Hilbert space, for example in a 2 qubit system we get the basis
        of a zero and one ket.
        A qubit is unique in that it can be in a superposition of the different states allowing for
        an increased number of values being represented
        Differs from a vector in that the contructor argument is simply positions of ones
        Args:
            bit_positions: A list of positions of ones in the quantum register
        """
        bit_pos_shape = np.shape(bit_positions)


        #Call the constructor of the vector class for a one'd sparse matrix at the specified bit
        #positions
        super(scipy.sparse.coo_matrix((np.ones(bit_pos_shape,\
                                      (bit_positions, np.zeros(bit_pos_shape))))))





















