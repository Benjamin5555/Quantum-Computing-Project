"""This module details the main and specific functionality of the circuit model including 
a functionality that will 'run' a circuit

Author(s): 
 * Benjamin Carpenter(S1731178@ed.ac.uk)
"""

from circuit_model import matrices
import scipy.sparse
import numpy as np

class Gate(matrices.SquareMatrix):
    """A class to implement gates, will share a large amount of functionality with a square matrix
    but will require have a more specific criteria for what form a matrix can take as well as
    likely requirng additional functionality
    
    Attributes:

    """
    def __init__(self, values):
        """Creates a gate object

        Args:

        Raises:
            ValueError: On invalid form for a Quantum circuit gate

        """
        #assert Hermatian, other requirments for a gate here
       
        #Call the parent class constructor after verifying valid parameters for a gate  
        NotImplemented##

class QuantumRegister(matrices.Vector):
    """A vector like object that represents a quantum register

    Attributes:

    """
    def __init__(self, bit_positions):
        """Creates a quantum register object.
        
        A quantum bit (qubit) is a system that can be observed in two unique states such as electron
        spin (can be up or down).

        A qubit is a vector in a Hilbert space (n-dimensional vector space) where \\(n\\) quibits 
        will represent \\(2^n\\) dimensional Hilbert space. 

        Each state is a basis of the Hilbert space, for example in a 2 qubit system we get the basis
        states $$\\bra{00}$$
         
        
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
                
class QuantumCircuit():
    """An object representing a quantum circuit, can be used to run circuits on ...
    
    
    """
    def __init__(self):
        NotImplemented
