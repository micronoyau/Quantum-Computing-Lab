import numpy as np
from math import *
import matplotlib.pyplot as plt
import random as rd


def dec2bin(n, size):
    """
    Returns binary value of n
    """
    l = []
    while n != 0:
        l.append(n % 2)
        n //= 2
    l = l + (size-len(l)) * [0]
    return l[::-1]

############################### def des qbit et des operateur ######################


ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])


H = np.array([[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, 1]])

CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
'''
CX = np.array([
    [[1, 0], [0, 0]],
    [[0, 1], [0, 0]],
    [[0, 0], [0, 1]],
    [[0, 0], [1, 0]]
])
'''
#########################################################################################


##################################### Time for change ###################################

class circuit:

    def __init__(self, N, khi):
        # all qbit start as 0

        self.N = N
        self.khi = khi
        self.MPS = list()

        # on initialise et termine avec des qbits virtuel pour ne pas avoir a utiliser de tenseur de dim 2

        self.MPS.append(np.zeros((2, khi), dtype=complex))

        for i in range(N):
            self.MPS.append(np.zeros((2, khi, khi), dtype=complex))

        self.MPS.append(np.zeros((2, khi), dtype=complex))

        self.MPS[0][0][0] = 1

        for M in self.MPS[1:-1]:
            M[0][0][0] = 1

        self.MPS[-1][0][0] = 1

    H = np.array([[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, 1]])

    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    def one_Qbit_gate(self, gate, qbit):
        self.MPS[qbit] = np.einsum('ij,jkl', gate, self.MPS[qbit])

    def rot_gate(self, theta, phi, psi):
        U = np.array([[cos(theta/2), -e**(complex(0, psi))*sin(theta/2)],
                      [e**(complex(0, phi))*sin(theta/2), e**(complex(0, psi + phi))*cos(theta/2)]])
        return U

    def resultat(self):
        tens = self.MPS[0]

        for i in range(1, self.N+1):
            tens = np.einsum(tens, list(range(1, i+2)),
                             self.MPS[i], [0, i+2, i+1])

        tens = np.einsum(tens, list(range(1, self.N+3)),
                         self.MPS[-1], [0, self.N+2])

        ket = np.zeros(2**self.N, dtype=complex)

        for i in range(2**self.N):
            binstring = dec2bin(i, self.N)
            sub = tens[0]
            for k in binstring:
                sub = sub[k]
            ket[i] = sub[0]

        return ket

    def state(self):
        tens = self.MPS[0]

        for i in range(1, self.N+1):
            tens = np.einsum(tens, list(range(1, i+2)),
                             self.MPS[i], [0, i+2, i+1])

        tens = np.einsum(tens, list(range(1, self.N+3)),
                         self.MPS[-1], [0, self.N+2])
        return tens

    def swap(self, qbit1, qbit2):
        self.MPS[qbit1], self.MPS[qbit2] = self.MPS[qbit2], self.MPS[qbit1]
        return

    def two_Qbit_gate(self, gate, qbit, controle_bit):

        gate_ord4 = np.zeros((2, 2, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        gate_ord4[i, j, k, l] = gate[2*i+j, 2*k+l]
#        switcharoo = False
        if qbit <= self.N:
            self.swap(qbit+1, controle_bit)
            pair = np.einsum('ijk,lkm', self.MPS[qbit+1], self.MPS[qbit])
            pair = np.einsum('ijkl,kmln', gate_ord4, pair)
#       else:
#           switcharoo = True
#           self.swap(qbit, qbit-1)
#           self.swap(qbit, controle_bit)
#           qbit = qbit-1
#           pair = np.einsum('ijk,lkm', self.MPS[qbit], self.MPS[qbit+1])
#           pair = np.einsum('iklj,kmln', gate_ord4, pair)

        mat_pair = np.zeros((2*self.khi, 2*self.khi), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(self.khi):
                    for l in range(self.khi):
                        mat_pair[i*self.khi + k, j *
                                 self.khi + l] = pair[i, j, k, l]

        ###SVD###
        X_loc, S_loc, Y_loc = np.linalg.svd(mat_pair)

        S_trunc = S_loc[:self.khi]

        ###returning the result to the circuit###
        # on troncate directement X à sa création pour gagner en temps de calcul
        X_tens = np.zeros((2, self.khi, self.khi), dtype=complex)
        for i in range(2):
            for j in range(self.khi):
                for k in range(self.khi):
                    X_tens[i, j, k] = X_loc[i*self.khi + j, k]

        # on troncate directement Y à sa création pour gagner en temps de calcul
        Y_tens = np.zeros((2, self.khi, self.khi), dtype=complex)
        for i in range(2):
            for j in range(self.khi):
                for k in range(self.khi):
                    Y_tens[i, j, k] = Y_loc[j, i*self.khi + k]

        for i in range(2):
            for j in range(self.khi):
                for k in range(self.khi):
                    self.MPS[qbit][i, j, k] = X_tens[i, j, k]*S_trunc[k]

        self.MPS[qbit+1] = Y_tens

        self.swap(qbit+1, controle_bit)
#       if switcharoo:
#           self.swap(qbit, qbit+1)
        return

    def two_Qbit_gate_fidelity(self, gate, qbit, controle_bit):

        gate_ord4 = np.zeros((2, 2, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        gate_ord4[i, j, k, l] = gate[2*i+j, 2*k+l]

        if qbit <= self.N:
            self.swap(qbit+1, controle_bit)
            pair = np.einsum('ijk,lkm', self.MPS[qbit+1], self.MPS[qbit])
            pair = np.einsum('ijkl,kmln', gate_ord4, pair)

        mat_pair = np.zeros((2*self.khi, 2*self.khi), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(self.khi):
                    for l in range(self.khi):
                        mat_pair[i*self.khi + k, j *
                                 self.khi + l] = pair[i, j, k, l]

        ###SVD###
        X_loc, S_loc, Y_loc = np.linalg.svd(mat_pair)

        f = (sum([i**2 for i in S_loc[:self.khi]]) /
             sum([i**2 for i in S_loc]))

        S_trunc = S_loc[:self.khi]

        ###returning the result to the circuit###
        # on troncate directement X à sa création pour gagner en temps de calcul
        X_tens = np.zeros((2, self.khi, self.khi), dtype=complex)
        for i in range(2):
            for j in range(self.khi):
                for k in range(self.khi):
                    X_tens[i, j, k] = X_loc[i*self.khi + j, k]

        # on troncate directement Y à sa création pour gagner en temps de calcul
        Y_tens = np.zeros((2, self.khi, self.khi), dtype=complex)
        for i in range(2):
            for j in range(self.khi):
                for k in range(self.khi):
                    Y_tens[i, j, k] = Y_loc[j, i*self.khi + k]

        for i in range(2):
            for j in range(self.khi):
                for k in range(self.khi):
                    self.MPS[qbit][i, j, k] = X_tens[i, j, k]*S_trunc[k]

        self.MPS[qbit+1] = Y_tens

        self.swap(qbit+1, controle_bit)

        return f

    def histo_test(self, taille):
        res = self.resultat()
        mod_res = [abs(i)**2 for i in res]
        proba = [0] + mod_res + [1]
        test = []
        for i in range(taille):
            p = rd.random()
            j = 0
            som = 0
            while p > som:
                j = j + 1
                som += proba[j]
            test.append(j-1)
        plt.hist(test)
        plt.show()
        return

    def benchmark(self, D):

        F = [1]
        F_av = [1]

        ## setup du circuit ##
        for i in range(D):

            for qbit in range(1, self.N):
                alpha = 2 * np.pi * np.random.random()
                phi = 2 * np.pi * np.random.random()
                theta = 2 * np.pi * np.random.random()
                self.one_Qbit_gate(self.rot_gate(alpha, phi, theta), qbit)

            F.append(1)
            F_av.append(F_av[-1])

            for qbit in range(1, self.N):
                if qbit % 2 == 0:
                    controle_bit = qbit + 1
                    f = self.two_Qbit_gate_fidelity(
                        self.CZ, qbit, controle_bit)
                    # print(f)
                    F[-1] *= f
                    F_av[-1] *= f

            F[-1] = F[-1]**(1 / (self.N-1))

        for d in range(len(F_av)):
            F_av[d] = F_av[d]**(1 / ((d+1)*(self.N-1)))

        return F, F_av


'''
#########################################################################################
c = circuit(3, 2)

# print('MPS:' + str(c.MPS))
# print('resultat: ' + str(c.resultat()))

c.one_Qbit_gate(X, 2)
c.swap(3, 3)
c.two_Qbit_gate(CX, 1, 2)
print('resultat: ' + str(c.resultat()))
print(c.histo_test(1000))

#########################################################################################

c = circuit(2, 2)
c.one_Qbit_gate(X, 1)
c.one_Qbit_gate(X, 2)
c.one_Qbit_gate(H, 1)
c.one_Qbit_gate(H, 2)

c.one_Qbit_gate(X, 1)
c.two_Qbit_gate(CX, 2, 1)

c.one_Qbit_gate(H, 1)


print('Deutsch: ' + str(c.resultat()))

c.histo_test(1000)
'''

#########################################################################################
####################################   Benchmark    #####################################
#########################################################################################
D = 200
c = circuit(40, 64)
L_F, F_av = c.benchmark(D)
print('fidelity :' + str(L_F))
L_D = [i for i in range(D + 1)]

plt.plot(L_D, L_F, label='seq')
plt.plot(L_D, F_av, label='av')
plt.legend()
plt.show()

