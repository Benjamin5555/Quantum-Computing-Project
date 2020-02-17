udlass Gate(matrices.SquareMatrix):
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


