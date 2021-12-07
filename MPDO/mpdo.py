import numpy as np
from copy import deepcopy

"""
MPDO quantum circuit simulator.
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


class MPDO:

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
    ######################       General MPDO pure state operations        ###################
    ##########################################################################################
    ##########################################################################################

    def __init__ (self, N, khi, kappa):
        """
        [N] is the number of qbits
        [khi] is the maximum entanglement degree
        [kappa] is the maximum statistical dispersion
        At the beginning, all qbits are 0s
        """
        self.N = N
        self.khi = khi
        self.kappa = kappa
        # The structure is the following
        #
        #  |    |    |     |    |    |
        #  o----o----o ... o----o----o  (up mpos)
        #  |    |    |     |    |    |
        #  o----o----o ... o----o----o  (down mpos : conjugate of up mpos)
        #  |    |    |     |    |    |
        #
        self.mpo = list () # List of upper matrix product operators

        # First tensor of rank 3 (first qbit, virtual, not used)
        self.mpo.append ( np.zeros((2, kappa, khi), dtype=complex) )

        # Tensors of rank 4 (qbits in the middle, used)
        for i in range (N):
            self.mpo.append ( np.zeros((2, kappa, khi, khi), dtype=complex) )

        # Last tensor of rank 3 (last qbit, virtual, not used)
        self.mpo.append ( np.zeros((2, kappa, khi), dtype=complex) )

        # One possible MPO state for representing |0><0|
        self.mpo[0][0][0][0] = 1
        for tensor in self.mpo[1:-1]:
            tensor[0][0][0][0] = 1
        self.mpo[-1][0][0][0] = 1


    def null_state (self):
        """
        Go back to initial state
        """
        self.__init__ (self.N, self.khi, self.kappa)


    def gate_1qbit (self, U, qbit):
        """
        Apply a 1qbit gate
        [qbit] is the qbit on which we apply the 1 qbit gate
        [U] is the unitary matrix corresponding to the gate : np.array
        """
        assert qbit >=0 and qbit < self.N, "Provided qbit index is out of bounds !"
        # Efficient tensor contraction
        self.mpo[qbit+1] = np.einsum('ij,jklm', U, self.mpo[qbit+1])


    def right_canonical_form (self, qbit):
        """
        Puts the MPDO in right canonical form up to qbit [qbit] (excluded, so [qbit]-1 included)
        """
        # We start from right (A(1)) and stop at qbit-1
        for l in range (1, qbit):
            # Reshape tensor into matrix
            A_tilde = np.asmatrix (np.zeros ((self.khi, 2*self.kappa*self.khi), dtype=complex))
            for il in range (2):
                for al in range (self.kappa):
                    for mul in range (self.khi):
                        for mul_1 in range (self.khi):
                            A_tilde[mul, il*self.kappa*self.khi + al*self.khi + mul_1] = self.mpo[l][il][al][mul][mul_1]

            # QR decomposition
            q, r = np.linalg.qr (A_tilde.getH())

            # Dropping useless parts (zeros and multiplied by zeros)
            r = r[:self.khi,:]
            q = q[:,:self.khi]

            # Getting dagger operators (need for right canonical form)
            q = q.getH()
            r = r.getH()

            # Update MPO
            for il in range (2):
                for al in range (self.kappa):
                    for mul in range (self.khi):
                        for mul_1 in range (self.khi):
                            self.mpo[l][il][al][mul][mul_1] = q[mul, il*self.kappa*self.khi + al*self.khi + mul_1]

            self.mpo[l+1] = np.einsum ('ijkl,lm', self.mpo[l+1], r)


    def left_canonical_form (self, qbit):
        """
        Puts the MPDO in left canonical form up to qbit [qbit] (excluded, so [qbit]+1 included)
        """
        # We start from left (A(N-2)) and stop at qbit-1
        for l in range (self.N, qbit, -1):
            # Reshape tensor into matrix
            A_tilde = np.asmatrix (np.zeros ((2*self.kappa*self.khi, self.khi), dtype=complex))
            for il in range (2):
                for al in range (self.kappa):
                    for mul in range (self.khi):
                        for mul_1 in range (self.khi):
                            A_tilde[il*self.kappa*self.khi + al*self.khi + mul, mul_1] = self.mpo[l][il][al][mul][mul_1]

            # QR decomposition
            q, r = np.linalg.qr (A_tilde)

            # Dropping useless parts (zeros and multiplied by zeros)
            r = r[:self.khi,:]
            q = q[:,:self.khi]

            # Update MPDO
            for i_l in range (2):
                for ip_l in range (self.kappa):
                    for mu_l in range (self.khi):
                        for mu_l_1 in range (self.khi):
                            self.mpo[l][il][al][mul][mul_1] = q[il*self.kappa*self.khi + al*self.khi + mul, mul_1]

            self.mpo[l-1] = np.einsum ('km,ijml', r, self.mpo[l-1])


    def canonical_form (self, qbit):
        """
        Puts the MPDO in canonical form centered on qbits [qbit] and [qbit] + 1
        through a series of QR decompositions.
        """
        self.left_canonical_form (qbit+1)
        self.right_canonical_form (qbit)


    def gate_2qbit_adj (self, U, qbit):
        """
        Apply unitary 2 qbit gate on adjacent qbits [qbit] and [qbit]+1
        """
        assert qbit >= 0 and qbit < self.N-1, "Provided qbit index is out of bounds !"

        # Step 1 : canonical form
        self.canonical_form (qbit)

        # Step 2
        T = np.einsum('ikmn,jlno', self.mpo[qbit+2], self.mpo[qbit+1])

        # Step 3
        U_tilde = np.zeros((2,2,2,2), dtype=complex)
        for a in range (2):
            for b in range (2):
                for c in range (2):
                    for d in range (2):
                        U_tilde[a][b][c][d] = U[2*a+b][2*c+d]
        Tp = np.einsum('ijkl,klmnop', U_tilde, T)

        # Step 4

        # Reshape Tp into matrix of size 2*khi by 2*khi,
        # following special conventions in paper
        Tp_tilde = np.zeros((4*self.khi, 4*self.khi), dtype=complex)
        for ik in range (2):
            for ipk in range (2):
                for ik_1 in range (2):
                    for ipk_1 in range (2):
                        for muk in range (self.khi):
                            for muk_2 in range (self.khi):
                                Tp_tilde[ik*2*self.khi + ipk*self.khi + muk][ik_1*2*self.khi + ipk_1*self.khi + muk_2] = Tp[ik][ik_1][ipk][ipk_1][muk][muk_2]

        # SVD
        X_tilde, S, Y_tilde = np.linalg.svd (Tp_tilde)

        # Truncate S
        S = S[:self.khi]

        # Back to tensors and truncate X and Y
        X = np.zeros((2, 2, self.khi, self.khi), dtype=complex)
        for ik in range (2):
            for ipk in range (2):
                for muk in range (self.khi):
                    for muk_1 in range (self.khi):
                        X[ik][ipk][muk][muk_1] = X_tilde[ik*2*self.khi + ipk*self.khi + muk][muk_1]

        Y = np.zeros((2, 2, self.khi, self.khi), dtype=complex)
        for ik_1 in range (2):
            for ipk_1 in range (2):
                for muk_1 in range (self.khi):
                    for muk_2 in range (self.khi):
                        Y[ik_1][ipk_1][muk_1][muk_2] = Y_tilde[muk_1][ik_1*2*self.khi + ipk_1*self.khi + muk_2]

        # Step 5

        # Contraction of X and S
        for ik in range (2):
            for ipk in range (2):
                for muk in range (self.khi):
                    for muk_1 in range (self.khi):
                        self.mpo[qbit+2][ik][ipk][muk][muk_1] = X[ik][ipk][muk][muk_1] * S[muk_1]
        self.mpo[qbit+1] = Y


    ##########################################################################################
    ##########################################################################################
    ###################       Arbitrary gates for non-neighbour qbits       ##################
    ##########################################################################################
    ##########################################################################################

    def swap_adj (self, qbit):
        """
        SWAP two adjacent qbits [qbit] and [qbit]+1
        """
        self.gate_2qbit_adj (MPDO.SWAP, qbit)


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
        self.gate_1qbit (MPDO.X, qbit)

    def y (self, qbit):
        self.gate_1qbit (MPDO.Y, qbit)

    def z (self, qbit):
        self.gate_1qbit (MPDO.Z, qbit)

    def h (self, qbit):
        self.gate_1qbit (MPDO.H, qbit)

    def s (self, qbit):
        self.gate_1qbit (MPDO.S, qbit)

    def t (self, qbit):
        self.gate_1qbit (MPDO.T, qbit)


    @staticmethod
    def rot_unitary (alpha, phi, theta):
        """
        Returns the unitary for the rotation of angle [theta]
        around unit vector n([alpha], [phi]) in spherical coordinates
        """
        U = np.cos(theta/2) * MPDO.I - 1j * np.sin(theta/2) \
            * ( np.sin(alpha) * np.cos(phi) * MPDO.X \
            + np.sin(alpha) * np.sin(phi) * MPDO.Y \
            + np.cos(alpha) * MPDO.Z )
        return U

    def rot (self, alpha, phi, theta, qbit):
        """
        Apply a rotation of angle [theta] around unit vector n(alpha, phi) in spherical coordinates
        """
        U = MPDO.rot_unitary (alpha, phi, theta)
        self.gate_1qbit (U, qbit)

    def rz (self, theta, qbit):
        self.rot (0, 0, theta, qbit)


    ##########################################################################################
    ##########################################################################################
    ###############################     Usual 2 qbit gates        ############################
    ##########################################################################################
    ##########################################################################################


    def cx (self, control, target):
        self.controlled_gate (MPDO.X, control, target)

    def cz (self, control, target):
        self.controlled_gate (MPDO.Z, control, target)

    def crot (self, alpha, phi, theta, control, target):
        """
        Apply a controlled rotation of angle [theta] around unit vector n(alpha, phi) in spherical coordinates
        """
        U = MPDO.rot_unitary (alpha, phi, theta)
        self.controlled_gate (U, control, target)

    def cphi (self, phi, control, target):
        """
        Apply a controlled phase shift of angle [phi]
        """
        U = MPDO.rot_unitary (0, 0, phi)
        U = np.exp (1j * phi / 2) * U
        self.controlled_gate (U, control, target)

    def toffoli (self, control1, control2, target):
        """
        Apply a double controlled NOT gate (Toffoli gate)
        """
        SQRT_X = np.exp(1j * np.pi / 4) * MPDO.rot_unitary (np.pi/2, 0, np.pi / 2)
        SQRT_X_DG = np.exp(- 1j * np.pi / 4) * MPDO.rot_unitary (np.pi/2, 0, - np.pi / 2)

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
    ##############################       Superoperators and noise     ########################
    ##########################################################################################
    ##########################################################################################

    def kraus (self, E):
        """
        [E] is the list of Kraus operators that describe the superoperator to apply.
        e (rho) = sum E rho E^dagger
        Each E_k in E can either be a one qbit gate represented by
        E_k = (c, U, qbit)   with U the unitary, c the coefficient in front of U
        or a 2 qbit gate represented by
        E_k = (U, qbit1, qbit2)     with U the 2 qbit unitary
        """
        # Make a copy of the whole MPO
        mpo_copy = deepcopy (self.mpo)

        # The sum of MPOs before we truncate inner indices
        final = list(np.zeros((self.N,2,0,self.khi,self.khi), dtype=complex))

        for E_k in E:
            # 1 qbit gate
            if len (E_k) == 3:
                c = E_k[0]
                U = E_k[1]
                qbit = E_k[2]

                # Apply 1 qbit gate
                self.gate_1qbit (c*U, qbit)

                for q in range (self.N):
                    if q != qbit:
                        self.gate_1qbit (c*MPDO.I, q)


            # 2 qbit gate : TODO not working yet
            elif len (E_k) == 4:
                c = E_k[0]
                U = E_k[1]
                qbit1 = E_k[2]
                qbit2 = E_k[3]
                qbit1, qbit2 = min(qbit1, qbit2), max(qbit1, qbit2)

                # Apply 2 qbit gate
                self.gate_2qbit (c*U, qbit1, qbit2)

                for q in range (self.N):
                    if q != qbit1 and q != qbit2:
                        self.gate_1qbit (c*MPDO.I, q)


            # Add inner indices to every qbit
            for q in range (self.N):
                new_shape = list(final[q].shape)
                new_shape[1] += self.kappa
                new_final_qbit = np.zeros (new_shape, dtype=complex)
                new_final_qbit[:,:final[q].shape[1],:,:] = final[q]
                new_final_qbit[:,final[q].shape[1]:,:,:] = self.mpo[q+1]
                final[q] = new_final_qbit

            # Resetting to initial state
            self.mpo = deepcopy(mpo_copy)


        """
        print (final[1][1,0,0,0])
        print (final[0][0,0,0,0])
        print (final[1][0,self.kappa,0,0])
        print (final[0][0,self.kappa,0,0])
        print (final[1][0,2*self.kappa,0,0])
        print (final[0][1,2*self.kappa,0,0])
        """

        for k in range (self.N):
            # Statistical dispersion
            M = np.asmatrix (np.zeros((2*self.khi*self.khi, final[k].shape[1]), dtype=complex))
            for ik in range (2):
                for ak in range (final[k].shape[1]):
                    for muk in range (self.khi):
                        for muk_1 in range (self.khi):
                            M[ik * self.khi*self.khi + muk*self.khi + muk_1, ak] = final[k][ik, ak, muk, muk_1]

            # SVD
            X, S, Y = np.linalg.svd (M)
            print (S)

            # Keep only kappa highest singular values
            for ik in range (2):
                for ak in range (self.kappa):
                    for muk in range (self.khi):
                        for muk_1 in range (self.khi):
                            final[k][ik, ak, muk, muk_1] = S[ak] * X[ik * self.khi*self.khi + muk*self.khi + muk_1, ak]

            final[k] = final[k][:,:self.kappa,:,:]

        self.mpo[1:-1] = final

    ##########################################################################################
    ##########################################################################################
    ##############################        Outcome probabilities       ########################
    ##########################################################################################
    ##########################################################################################

    def get_probabilities (self):
        """
        Computes the density operator of the circuit state using the following steps:
        1) Compute tensor contraction of the MPDO
        2) Add up amplitudes according to tensor values
        """
        # Tensor contraction of the upper MPO : starting from the right
        tens = self.mpo[0]

        for i in range(1, self.N+1):
            tens = np.einsum (tens, list(range(1,i+1)) + list(range(i+2, 2*i+2)) + [2*(i+1),], self.mpo[i], [0, i+1, 2*(i+1), 2*i+3])

        # Last site to the left
        tens = np.einsum (tens, list(range(1, self.N+2)) + list(range(self.N+3, 2*self.N+4)) + [2*(self.N+2),], self.mpo[-1], [0, self.N+2, 2*(self.N+2)])

        # Then contract with its conjugate
        conj = np.conjugate (tens)
        tens = np.einsum (tens, list(range(2*self.N+4)), conj, list(range(2*self.N+4, 3*self.N+6)) + list(range(self.N+2, 2*self.N+4)))

        # Probabilities
        prob = np.zeros (2**self.N)

        for i in range( 2**self.N ):
            # Neat hack to write down prob[i] = tens[i_{N-1}]...[i_1][i_0]
            i_bin = dec2bin( i, self.N )
            # i
            sub = tens[0]
            for j in i_bin:
                sub = sub[j]
            sub = sub[0]
            # i'
            sub = sub[0]
            for j in i_bin:
                sub = sub[j]
            sub = sub[0]
            # Finally we have the probability
            prob[i] = sub

        return prob


    def __str__ (self):
        return str (self.get_probabilities())


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

        # Individual probabilities
        distribution = list(map(lambda x : abs(x)**2, ket))

        cumul_distrib = [distribution[0]]
        for k in range (1, len(distribution)):
            cumul_distrib.append( cumul_distrib[-1] + distribution[k] )

        # Sometimes not equal to 1 because of SVD approximation
        total = cumul_distrib[-1]
        # Random number to represent the outcom
        rand = np.random.random () * total

        for j in range (len(cumul_distrib)):
            if rand <= cumul_distrib[j]:
                self.null_state ()

                qbits = dec2bin (j, self.N)[::-1]

                for qbit in range (len(qbits)):
                    if qbits[qbit] == 1:
                        self.x (qbit)

                return j


    def measure (self, qbit):
        """
        Measure a single qbit, and leave the state in the collapsed state
        """
        ket = self.get_ket ()

        # Computing probability to get a |0>
        p0 = 0
        for i in range (pow(2, self.N-1)):
            lsb = i & (pow(2, qbit)-1)
            msb = i & (~lsb)
            msb <<= 1
            p0 += abs(ket[msb | lsb])**2

        # Sometimes not equal to 1 because of SVD approximation
        total = sum(map(lambda x:abs(x)**2, ket))
        rand = np.random.random () * total

        # The outcome is |0>
        if rand < p0:
            # We have to set the collapsed state
            # -> We project the current ket onto the eigenspace with projector (I+Z_qbit)/2
            # and we normalize it. This is equivalent to applying (I+Z_qbit) / (2*sqrt(p0))
            U = np.array ([[1.0, 0.0], [0.0, 0.0]])
            U /= np.sqrt(p0)
            self.gate_1qbit (U, qbit)
            return 0

        # The outcome is |1>
        U = np.array ([[0.0, 0.0], [0.0, 1.0]])
        U /= np.sqrt(1-p0)
        self.gate_1qbit (U, qbit)
        return 1
