"""
Implementation of Shor's factoring algorithm
with MPS representation
"""
from mps import QuantumComputer
import numpy as np
from abc import ABC, abstractmethod

np.set_printoptions (precision=3, suppress=True)

def gcd (a, b):
    assert a >= b
    while b != 0:
        a, b = b, a%b
    return a

class ShorComputer (QuantumComputer, ABC):

    def __init__ (self, khi, n):
        self.n = n # Composite number to factor
        self.d = int(np.log(n) / np.log(2)) + 1 # Size of first register
        self.t = 2 * int (np.log(n)/np.log(2) + 1) + 1 # Precision (size of second register)
        QuantumComputer.__init__ (self, self.t + self.d, khi)


    def v (self, i, j):
        self.cphi (np.pi / (2**abs(i-j)), i, j)

    def v_dg (self, i, j):
        self.cphi (- np.pi / (2**abs(i-j)), i, j)

    def QFT (self, q0, q1):
        """
        Apply QFT to qbits between [q0] and [q1], with q0 < q1
        """
        assert q0 < q1

        # First swap qbits : |x_{n-1} ... x_0 > ----> |x_0 ... x_{n-1} >
        for i in range ((q1 - q0 + 1)//2):
            self.swap (q0+i, q1-i)

        # Apply V_{i,j}
        for i in range (q0, q1+1):
            self.h (i)
            for j in range (i+1, q1+1):
                self.v (i,j)

    def QFT_dg (self, q0, q1):
        """
        Apply inverse QFT to qbits between [q0] and [q1], with q0 < q1
        """
        assert q0 < q1

        # Apply V_{i,j}^\dagger
        for i in range (q1, q0-1, -1):
            for j in range (q1, i, -1):
                self.v_dg (i,j)
            self.h (i)

        # Swap qbits : |x_{n-1} ... x_0 > ----> |x_0 ... x_{n-1} >
        for i in range ((q1 - q0 + 1)//2):
            self.swap (q0+i, q1-i)


    def cswap (self, control, target1, target2):
        self.toffoli (control, target1, target2)
        self.toffoli (control, target2, target1)
        self.toffoli (control, target1, target2)


    def prepare_eigenstate (self):
        # Last qbit : initializing second register to |1>
        self.x (self.t + self.d - 1)

    @abstractmethod
    def CU (self, a, k, control):
        """
        [a] is the element in (Z/nZ)* which we want to find the order
        For this reason, we must have gcd (a,n) = 1
        [k] is the exponent in the execution of U^{2^k}
        """
        pass


    def QPE (self, a):
        """
        We assess that the first [t] qbits are |0>, with [t] the precision wanted for the eigenvalue,
        and that the next [n] qbits contain the eigenstate |u> of [U]
        We then perform a QPE to estimate the phase of eigenvalue associated with |u>.

        [CU] is an instance method, that represents the controlled unitary U, which takes the following parameters :
         - [a] : element in (Z/nZ)* which we want to find the order
         - [k] : we apply U 2^k times to qbits [self.t] to [self.t]+[self.n]-1
         - [control] : control qbit

        def U (self, a, k, control):
            # Execute U^{2^k} controlled by [control]
            ...
        """
        # First apply Hadamards
        for i in range (self.t):
            self.h (i)

        # Implement |u> in second register
        self.prepare_eigenstate ()

        # Then apply U^{2^k} gates
        for k in range (self.t):
            self.CU (a, k, k)

        # Finally apply inverse QFT to gather eigenvalue
        self.QFT_dg (0, self.t-1)


    def run (self):
        """
        Core of Shor's algorithm.
        Executes the differents parts together.
        """
        # Step 1 : select a random a in [2, n-1]
        a = np.random.randint (2, self.n)
        print ("Trying with a = {}".format(a))

        # Step 2 : is a coprime with n ?
        if gcd (self.n, a) != 1:
            print ("Found prime factor ! (very lucky way) : p = {}".format(a))

        # Step 3 : compute order [r] of [a] modulo [n] with QPE
        self.QPE (a)


class Shor15 (ShorComputer):
    """
    Factor 15 with Shor's algorithm
    """
    def __init__ (self, khi):
        ShorComputer.__init__ (self, khi, 15)


    def CU (self, a, k, control):
        """
        Hard coded CU^2^k for [a] and [k], for n = 15
        """
        # Only elements which have gcd (a,n) = 1
        if a == 2:
            if k == 0:
                self.cswap (self.t+3, self.t+2, control)
                self.cswap (self.t+2, self.t+1, control)
                self.cswap (self.t+1, self.t, control)

            elif k == 1:
                self.cswap (self.t+2, self.t, control)
                self.cswap (self.t+3, self.t+1, control)

        elif a == 4:
            if k == 0:
                self.cswap (self.t+2, self.t, control)
                self.cswap (self.t+3, self.t+1, control)

        elif a == 7:
            if k == 0:
                self.cswap (self.t+1, self.t, control)
                self.cswap (self.t+2, self.t+1, control)
                self.cswap (self.t+3, self.t+2, control)
                for i in range (self.t, self.t+4):
                    self.cx (control, i)

            if k == 1:
                self.cswap (self.t+2, self.t, control)
                self.cswap (self.t+3, self.t+1, control)

        elif a == 8:
            if k == 0:
                self.cswap (self.t+1, self.t, control)
                self.cswap (self.t+2, self.t+1, control)
                self.cswap (self.t+3, self.t+2, control)

        elif a == 11:
            if k == 0:
                self.cswap (self.t+2, self.t, control)
                self.cswap (self.t+3, self.t+1, control)
                for i in range (self.t, self.t+4):
                    self.cx (control, i)

        elif a == 13:
            if k == 0:
                self.cswap (self.t+3, self.t+2, control)
                self.cswap (self.t+2, self.t+1, control)
                self.cswap (self.t+1, self.t, control)
                for i in range (self.t, self.t+4):
                    self.cx (control, i)

            if k == 1:
                self.cswap (self.t+2, self.t, control)
                self.cswap (self.t+3, self.t+1, control)

        elif a == 14:
            if k == 0:
                for i in range (self.t, self.t+4):
                    self.cx (control, i)


qc = Shor15 (16)
qc.run ()
