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
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    H = 1 / np.sqrt(2) * np.array([[1,1],[1,-1]])
    S = np.array([[1,0],[0,1j]])
    T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

    def __init__ (self, N, khi):
        """
        [N] is the number of qbits
        [khi] is the maximum entanglement degree (here lies the approximation)
        At the beginning, all qbits are 0s
        """
        self.N = N
        self.khi = khi
        self.mps = list() # List of tensors

        # First tensor of rank 2 (first qbit)
        self.mps.append ( np.zeros((2, khi)) )
        # Tensors of rank 3 (qbits in the middle)
        for i in range ( N-2 ):
            self.mps.append ( np.zeros((2, khi, khi)) )
        # Last tensor of rank 2
        self.mps.append ( np.zeros((2, khi)) )

        # One possible MPS state for representing |0>
        self.mps[0][0][0] = 1
        for tensor in self.mps[1:-1]:
            tensor[0][0][0] = 1
        self.mps[-1][0][0] = 1


    def gate_1qbit (self, U, qbit):
        """
        Apply a 1qbit gate
        [qbit] is the qbit on which we apply the 1 qbit gate
        [U] is the unitary matrix corresponding to the gate : np.array
        """
        # If only 2 indices
        if (qbit == 0 or qbit == self.N-1):
            # Efficient tensor contraction
            self.mps[qbit] = np.einsum('ij,jk', U, self.mps[qbit])

        else:
            # Tensor contraction
            self.mps[qbit] = np.einsum('ij,jkl', U, self.mps[qbit])


    def gate_2qbit_adj (self, U, qbit):
        """
        Apply unitary 2 qbit gate on adjacent qbits [qbit] and [qbit]+1
        """
        # Step 1
        T = np.einsum('ikl,jlm', self.mps[qbit+1], self.mps[qbit])

        # Step 2
        U_tilde = np.zeros((2,2,2,2))
        for a in range (2):
            for b in range (2):
                for c in range (2):
                    for d in range (2):
                        U_tilde[a][b][c][d] = U[2*a+b][2*c+d]
        Tp = np.einsum('ijkl,klmn', U_tilde, T)

        # Step 3

        # Reshape Tp intro matrix of size 2*khi by 2*khi,
        # following special conventions in paper
        Tp_tilde = np.zeros((2*self.khi, 2*self.khi))
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
        X = np.zeros((2, self.khi, 2*self.khi))
        for i in range (2):
            for k in range (self.khi):
                for l in range (2*self.khi):
                    X[i][k][l] = X_tilde[i*self.khi + k][l]

        Y = np.zeros((2, 2*self.khi, self.khi))
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
                    self.mps[qbit+1][i][k][l] = X[i][k][l] * S[l]

        self.mps[qbit] = Y


    def get_ket (self):
        """
        Computes the ket representation of the circuit using the following steps:
        1) Computes tensor contraction of the MPS
        2) Adds up amplitudes according to tensor values
        """
        # Tensor contraction of the MPS : starting from the right
        tens = self.mps[0]

        for i in range(1, self.N-1):
            tens = np.einsum (tens, list(range(1, i+2)), self.mps[i], [0, i+2, i+1])
            #tens = np.einsum (tens, list(range(i+1)), self.mps[i], [i+1,i,i+2])

        # Last site to the left
        #tens = np.einsum (tens, list(range(self.N)), self.mps[-1], [self.N, self.N-1])
        tens = np.einsum (tens, list(range(1, self.N+1)), self.mps[-1], [0, self.N])

        # Adding up values to the ket
        ket = np.array( [0.0] * (2**self.N) )

        for i in range( 2**self.N ):
            # Neat hack to write down ket[i] = tens[i_{N-1}]...[i_1][i_0]
            i_bin = dec2bin( i, self.N )
            sub = tens
            for j in i_bin:
                sub = sub[j]
            ket[i] = sub

        return ket 


    def __str__ (self):
        return str (self.get_ket())


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

    def swap (self, qbit):
        """
        SWAP two adjacent qbits [qbit] and [qbit]+1
        """
        self.gate_2qbit_adj (QuantumComputer.SWAP, qbit)


    def gate_2qbit (self, U, qbit1, qbit2):
        """
        Perform arbitrary 2 qbit gate [U] between qbits [qbit1] and [qbit2]
        """
        a = min(qbit1, qbit2)
        b = max(qbit1, qbit2)

        # First swap adjacent qbits to bring qbit1 near to qbit2
        for i in range (a, b-1):
            self.swap (i)

        # In case of non symmetric gates
        if qbit2 < qbit1:
            self.swap (b-1)

        self.gate_2qbit_adj (U, b-1)

        # Then unswap everything
        if qbit2 < qbit1:
            self.swap (b-1)

        for i in range (b-2, a-1, -1):
            self.swap(i)


    def cx (self, control, target):
        self.gate_2qbit (QuantumComputer.CNOT, target, control)

if __name__ == '__main__':
    qc = QuantumComputer (5, 3)

    qc.h(2)
    qc.cx(2,3)

    print(qc)
