import numpy as np

"""
MPS quantum circuit simulator.
"""

def dec2bin (n, size):
    """
    Returns binary value of n
    """
    l = []
    while n != 0:
        l.append(n%2)
        n //= 2
    l = l + (size-len(l)) * [0]
    return l[::-1]


class QuantumComputer :

    # Set of usual gates
    I = np.array([[1,0],[0,1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    H = 1 / np.sqrt(2) * np.array([[1,1],[1,-1]])
    S = np.array([[1,0],[0,1j]])
    T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
    SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

    ##########################################################################################
    ##########################################################################################
    ############################       General MPS operations        #########################
    ##########################################################################################
    ##########################################################################################

    def __init__ (self, N, khi):
        """
        [N] is the number of qbits
        [khi] is the maximum entanglement degree (here lies the approximation)
        At the beginning, all qbits are 0s
        """
        self.N = N
        self.khi = khi
        self.mps = list() # List of tensors

        # First tensor of rank 2 (first qbit, virtual, not used)
        self.mps.append ( np.zeros((2, khi), dtype=complex) )

        # Tensors of rank 3 (qbits in the middle, used)
        for i in range ( N ):
            self.mps.append ( np.zeros((2, khi, khi), dtype=complex) )

        # Last tensor of rank 2 (last qbit, virtual, not used)
        self.mps.append ( np.zeros((2, khi), dtype=complex) )

        # One possible MPS state for representing |0>
        self.mps[0][0][0] = 1
        for tensor in self.mps[1:-1]:
            tensor[0][0][0] = 1
        self.mps[-1][0][0] = 1


    def null_state (self):
        """
        Go back to initial state
        """
        self.__init__ (self.N, self.khi)


    def gate_1qbit (self, U, qbit):
        """
        Apply a 1qbit gate
        [qbit] is the qbit on which we apply the 1 qbit gate
        [U] is the unitary matrix corresponding to the gate : np.array
        """
        assert qbit >=0 and qbit < self.N, "Provided qbit index is out of bounds !"
        # Efficient tensor contraction
        self.mps[qbit+1] = np.einsum('ij,jkl', U, self.mps[qbit+1])


    def gate_2qbit_adj (self, U, qbit):
        """
        Apply unitary 2 qbit gate on adjacent qbits [qbit] and [qbit]+1
        """
        assert qbit >= 0 and qbit < self.N-1, "Provided qbit index is out of bounds !"

        # Step 1
        T = np.einsum('ikl,jlm', self.mps[qbit+2], self.mps[qbit+1])

        # Step 2
        U_tilde = np.zeros((2,2,2,2), dtype=complex)
        for a in range (2):
            for b in range (2):
                for c in range (2):
                    for d in range (2):
                        U_tilde[a][b][c][d] = U[2*a+b][2*c+d]
        Tp = np.einsum('ijkl,klmn', U_tilde, T)

        # Step 3

        # Reshape Tp intro matrix of size 2*khi by 2*khi,
        # following special conventions in paper
        Tp_tilde = np.zeros((2*self.khi, 2*self.khi), dtype=complex)
        for i in range (2):
            for j in range (2):
                for k in range (self.khi):
                    for l in range (self.khi):
                        Tp_tilde[i*self.khi + k][j*self.khi + l] = Tp[i][j][k][l]

        # SVD
        X_tilde, S, Y_tilde = np.linalg.svd (Tp_tilde)

        # Truncate S
        S = S[:self.khi]

        # Back to tensors
        X = np.zeros((2, self.khi, 2*self.khi), dtype=complex)
        for i in range (2):
            for k in range (self.khi):
                for l in range (2*self.khi):
                    X[i][k][l] = X_tilde[i*self.khi + k][l]

        Y = np.zeros((2, 2*self.khi, self.khi), dtype=complex)
        for i in range (2):
            for k in range (2*self.khi):
                for l in range (self.khi):
                    Y[i][k][l] = Y_tilde[k][i*self.khi + l]

        # Truncate X and Y
        X = X[:, :, :self.khi]
        Y = Y[:, :self.khi, :]

        # Step 4

        # Contraction of X and S
        for i in range (2):
            for k in range (self.khi):
                for l in range (self.khi):
                    self.mps[qbit+2][i][k][l] = X[i][k][l] * S[l]

        self.mps[qbit+1] = Y


    ##########################################################################################
    ##########################################################################################
    ###################       Arbitrary gates for non-neighbour qbits       ##################
    ##########################################################################################
    ##########################################################################################

    def swap_adj (self, qbit):
        """
        SWAP two adjacent qbits [qbit] and [qbit]+1
        """
        self.gate_2qbit_adj (QuantumComputer.SWAP, qbit)


    def swap (self, qbit1, qbit2):
        """
        SWAP two qbits [qbit1] and [qbit2]
        """
        a = min (qbit1, qbit2)
        b = max (qbit1, qbit2)

        # Bring a next to b, where a takes the place of b
        for i in range (a, b):
            self.swap_adj (i)

        # Bring b to the former place of a
        for i in range (b-2, a-1, -1):
            self.swap_adj (i)


    def gate_2qbit (self, U, qbit1, qbit2):
        """
        Perform arbitrary 2 qbit gate [U] between qbits [qbit1] and [qbit2]
        """
        a = min(qbit1, qbit2)
        b = max(qbit1, qbit2)

        # First swap adjacent qbits to bring qbit1 near to qbit2
        for i in range (a, b-1):
            self.swap_adj (i)

        # In case of non symmetric gates
        if qbit2 < qbit1:
            self.swap_adj (b-1)

        self.gate_2qbit_adj (U, b-1)

        # Then unswap everything
        if qbit2 < qbit1:
            self.swap_adj (b-1)

        for i in range (b-2, a-1, -1):
            self.swap_adj (i)



    def controlled_gate (self, U, control, target):
        """
        Build controlled U : apply U on [target] qbit if and only if [control] is |1>
        """
        gate = np.eye (4, dtype=complex)
        gate[2][2] = U[0][0]
        gate[2][3] = U[0][1]
        gate[3][2] = U[1][0]
        gate[3][3] = U[1][1]

        self.gate_2qbit (gate, target, control)


    ##########################################################################################
    ##########################################################################################
    ###############################     Usual 1 qbit gates        ############################
    ##########################################################################################
    ##########################################################################################

    def x (self, qbit):
        self.gate_1qbit (QuantumComputer.X, qbit)

    def y (self, qbit):
        self.gate_1qbit (QuantumComputer.Y, qbit)

    def z (self, qbit):
        self.gate_1qbit (QuantumComputer.Z, qbit)

    def h (self, qbit):
        self.gate_1qbit (QuantumComputer.H, qbit)

    def s (self, qbit):
        self.gate_1qbit (QuantumComputer.S, qbit)

    def t (self, qbit):
        self.gate_1qbit (QuantumComputer.T, qbit)


    @staticmethod
    def rot_unitary (alpha, phi, theta):
        """
        Returns the unitary for the rotation of anglaie [theta]
        around unit vector n([alpha], [phi]) in spherical coordinates
        """
        U = np.cos(theta/2) * QuantumComputer.I - 1j * np.sin(theta/2) \
            * ( np.sin(alpha) * np.cos(phi) * QuantumComputer.X \
            + np.sin(alpha) * np.sin(phi) * QuantumComputer.Y \
            + np.cos(alpha) * QuantumComputer.Z )
        return U

    def rot (self, alpha, phi, theta, qbit):
        """
        Apply a rotation of angle [theta] around unit vector n(alpha, phi) in spherical coordinates
        """
        U = QuantumComputer.rot_unitary (alpha, phi, theta)
        self.gate_1qbit (U, qbit)

    def rz (self, theta, qbit):
        self.rot (0, 0, theta, qbit)


    ##########################################################################################
    ##########################################################################################
    ###############################     Usual 2 qbit gates        ############################
    ##########################################################################################
    ##########################################################################################


    def cx (self, control, target):
        self.controlled_gate (QuantumComputer.X, control, target)

    def cz (self, control, target):
        self.controlled_gate (QuantumComputer.Z, control, target)

    def crot (self, alpha, phi, theta, control, target):
        """
        Apply a controlled rotation of angle [theta] around unit vector n(alpha, phi) in spherical coordinates
        """
        U = QuantumComputer.rot_unitary (alpha, phi, theta)
        self.controlled_gate (U, control, target)

    def cphi (self, phi, control, target):
        """
        Apply a controlled phase shift of angle [phi]
        """
        U = QuantumComputer.rot_unitary (0, 0, phi)
        U = np.exp (1j * phi / 2) * U
        self.controlled_gate (U, control, target)

    def toffoli (self, control1, control2, target):
        """
        Apply a double controlled NOT gate (Toffoli gate)
        """
        SQRT_X = np.exp(1j * np.pi / 4) * QuantumComputer.rot_unitary (np.pi/2, 0, np.pi / 2)
        SQRT_X_DG = np.exp(- 1j * np.pi / 4) * QuantumComputer.rot_unitary (np.pi/2, 0, - np.pi / 2)

        self.controlled_gate (SQRT_X, control1, target)
        self.cx (control2, control1)
        self.controlled_gate (SQRT_X_DG, control1, target)
        self.cx (control2, control1)
        self.controlled_gate (SQRT_X, control2, target)


    def cswap (self, control, target1, target2):
        self.toffoli (control, target1, target2)
        self.toffoli (control, target2, target1)
        self.toffoli (control, target1, target2)


    ##########################################################################################
    ##########################################################################################
    ##############################        Ket representation          ########################
    ##########################################################################################
    ##########################################################################################

    def get_ket (self):
        """
        Computes the ket representation of the circuit using the following steps:
        1) Computes tensor contraction of the MPS
        2) Adds up amplitudes according to tensor values
        """
        # Tensor contraction of the MPS : starting from the right
        tens = self.mps[0]

        for i in range(1, self.N+1):
            tens = np.einsum (tens, list(range(1, i+2)), self.mps[i], [0, i+2, i+1])

        # Last site to the left
        tens = np.einsum (tens, list(range(1, self.N+3)), self.mps[-1], [0, self.N+2])

        # Adding up values to the ket
        ket = np.array( [0.0] * (2**self.N), dtype=complex )

        for i in range( 2**self.N ):
            # Neat hack to write down ket[i] = tens[i_{N-1}]...[i_1][i_0]
            i_bin = dec2bin( i, self.N )
            sub = tens[0]
            for j in i_bin:
                sub = sub[j]
            sub = sub[0]
            ket[i] = sub

        return ket 


    def __str__ (self):
        return str (self.get_ket())


    ##########################################################################################
    ##########################################################################################
    #################################         Measurement         ############################
    ##########################################################################################
    ##########################################################################################

    def measure_all (self):
        """
        Measure all qbits, and leave the state in the collapsed state
        """
        ket = self.get_ket()
        # Theoritical distribution
        distribution = list(map(lambda x : abs(x)**2, ket))
        cumul_distrib = [distribution[0]]
        for k in range (1, len(distribution)):
            cumul_distrib.append( cumul_distrib[-1] + distribution[k] )

        # Sometimes not equal to 1 because of SVD approximation
        total = cumul_distrib[-1]
        rand = np.random.random () * total

        for j in range (len(cumul_distrib)):
            if rand <= cumul_distrib[j]:
                self.null_state ()

                qbits = dec2bin (j, self.N)[::-1]

                for qbit in range (len(qbits)):
                    if qbits[qbit] == 1:
                        self.x (qbit)

                return j
