from mps import QuantumComputer
from matplotlib import pyplot as plt
import numpy as np
import csv

"""
Measurement of the fidelity of MPS quantum circuit
"""

class BenchmarkQuantumComputer (QuantumComputer):
    """
    Measurement of the fidelity
    Any 2 qbit gate returns its fidelity
    """

    def gate_2qbit_adj (self, U, qbit):
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

        # Truncate S + record fidelity
        f = sum( [S[i]**2 for i in range(self.khi)] ) / sum( [s**2 for s in S] )
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

        return f # return fidelity of operation


    def cz_adj (self, qbit):
        return self.gate_2qbit_adj (QuantumComputer.CZ, qbit)



def benchmark_fidelity (N, D, khi):
    """
    Generates a random circuit of depth [D] with [N] qbits,
    according to paper, and with entanglement degree [khi]

    Returns quantum circuit [qc] and fidelity measures [f]
    """

    qc = BenchmarkQuantumComputer (N, khi)

    # fidelity after each 2 qbit gate
    f = []

    for d in range(D):
        # First apply 1 qbit gates (rotations) of angle theta around m(alpha, phi)
        for qbit in range (N):
            alpha = 2 * np.pi * np.random.random()
            phi = 2 * np.pi * np.random.random()
            theta = 2 * np.pi * np.random.random()
            qc.rot (alpha, phi, theta, qbit)

        # Then apply 2 qbit gates
        if d%2 == 0:
            for qbit in range (N//2):
                f.append (qc.cz_adj (2 * qbit))

        else:
            for qbit in range ((N-1)//2):
                f.append (qc.cz_adj (2 * qbit + 1))

        print (" --- Done with d =", d, "---")

    return f


def sequence_fidelity (N, D, f):
    """
    Postprocessing part : computes the fidelity after each sequence,
    that is the geometric average of each 2 qbit gate between
    depth [D]-2 and [D]
    """
    f_seq = [1,1]

    for d in range (2, D):
        f_seq.append (1)

        if d%2 == 0:
            for i in range (N-1):
                f_seq[-1] *= f[ (d//2 - 1) * (N-1) + i ]

        if d%2 == 1:
            for i in range (N-1):
                f_seq[-1] *= f[ (N//2) + ((d-1)//2 - 1) * (N-1) + i ]

        f_seq[-1] = pow (f_seq[-1], 1/(N-1))

    return f_seq


def average_fidelity (N, D, f):
    """
    Postprocessing part : computes the geometric average fidelity from the beginning,
    as a function of depth
    """
    f_av = [1]

    # First multiply (cumulative products)
    for d in range (D):

        if d%2 == 0:
            for i in range (N//2):
                f_av[-1] *= f[ (d//2) * (N-1) + i ]

        if d%2 == 1:
            for i in range ((N-1)//2):
                f_av[-1] *= f[ (d-1)//2 * (N-1) + N//2 + i ]

        f_av.append (f_av[-1])

    f_av.pop(-1)

    # Then add exponent
    for d in range (D):

        if d%2 == 0:
            f_av[d] = pow ( f_av[d], 1 / (N//2 + d//2 * (N-1)) )

        if d%2 == 1:
            f_av[d] = pow ( f_av[d], 1 / ((d+1)//2 * (N-1)) )

    return f_av


def plot_from_file (filename):
    """
    Plot data from file [filename]
    """

    with open (filename, 'r') as csvfile:
        reader = csv.reader (csvfile, delimiter = ' ')
        next(reader)
        next(reader)
        next(reader)
        next(reader)
        f_seq = list(map(float, next(reader)))
        next(reader)
        f_av = list(map(float, next(reader)))

        plt.plot(f_seq)
        plt.plot(f_av)
        plt.show()


def benchmark_save (N, D, khi, filename):
    f = benchmark_fidelity (N, D, khi)
    f_seq = sequence_fidelity (N, D, f)
    f_av = average_fidelity (N, D, f)

    with open (filename, 'w') as csvfile:
        writer = csv.writer (csvfile, delimiter=' ')
        writer.writerow (['N = {}'.format(N), 'D = {}'.format(D), 'khi = {}'.format(khi)])

        writer.writerow (['f'])
        writer.writerow (f)

        writer.writerow (['f_seq'])
        writer.writerow (f_seq)

        writer.writerow (['f_av'])
        writer.writerow (f_av)


np.set_printoptions(precision=5, suppress=True)

benchmark_save (40, 200, 64, 'CZ.csv')
plot_from_file ('CZ.csv')
