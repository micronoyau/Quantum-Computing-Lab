from qiskit import QuantumCircuit, assemble, Aer
import numpy as np
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

def gcd (a, b):
    if a < b:
        a, b = b, a
    while b != 0:
        a, b = b, a%b
    return a

def bin2dec (binary):
    # We assume the representation is a string, with MSB at the left
    ret = 0
    for i in range(len(binary)):
        ret += (int(binary[i]) << (len(binary)-i-1))
    return ret

def continuous_fraction_coefs (a, b):
    """
    Continuous fraction coefficients of a/b
    """
    ret = []
    while a != 0:
        ret.append( int(b//a) )
        a, b = b%a, a
    return ret

def partial_sum (coefs):
    """
    [coefs] has shape [a0, ..., an], and we compute 1/(a0 + 1/(a1 + ...) ) as p/q, with gcd(p,q) = 1
    """
    if len (coefs) == 1:
        return (1, coefs[0])

    p,q = partial_sum (coefs[1:])
    den = coefs[0] * q + p
    num = q
    g = gcd(p, q)
    return (num//g, den//g)


class ShorComputer (QuantumCircuit, ABC):

    def __init__ (self, n):
        self.n = n
        self.d = int (np.log(n) / np.log(2)) + 1 # Size of first register (eigenvector)
        self.t = 2 * int (np.log(n) / np.log(2) + 1) + 1 # Size of second register (precision)
        self.N = self.d + self.t
        QuantumCircuit.__init__ (self, self.N)

    def null_state (self):
        pass

    def v (self, i, j):
        # Controlled phase gate
        self.cp (np.pi / (2**abs(i-j)), i, j)

    def v_dg (self, i, j):
        self.cp (-np.pi / (2**abs(i-j)), i, j)

    def QFT (self, q0, q1):
        """
        Apply QFT to qbits between [q0] and [q1], with q0 < q1
        """
        assert q0 < q1

        # First swap qbits : | x_{n-1} ... x_0 >  ----->  | x_0 ... x_{n-1} >
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
                self.v_dg (i, j)
            self.h (i)

        # Swap qbits : | x_{n-1} ... x_0 >  ----->  | x_0 ... x_{n-1} >
        for i in range ((q1 - q0 + 1)//2):
            self.swap (q0+i, q1-i)


    def prepare_eigenstate (self):
        # Last qbit : initializing first register to |1> (superposition of eigenvectors)
        self.x (0)

    @abstractmethod
    def cu2k (self, a, k, control):
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
         - [k] : we apply U 2^k times to qbits 0 to [self.d]-1
         - [control] : control qbit

        def U (self, a, k, control):
            # Execute U^{2^k} controlled by [control]
            ...
        """
        # First apply Hadamards
        for i in range (self.t):
            self.h (i + self.d)

        # Implement |u> in second register
        self.prepare_eigenstate ()

        # Then apply U^{2^k} gates
        for k in range (self.t):
            self.cu2k (a, k, self.d+k)

        # Finally apply inverse QFT to gather eigenvalue
        self.QFT_dg (self.d, self.N-1)


    def run (self):
        """
        Core of Shor's algorithm.
        Executes the differents parts together.
        """
        # Step 1 : select a random a in [2, n-1]
        a = np.random.randint (2, self.n)
        print ("Trying with a = {}".format(a))

        # Step 2 : is a coprime with n ?
        g = gcd (self.n, a)
        if g != 1:
            print ("Found prime factor ! (very lucky way) : p = {}".format(g))
            return (g, self.n//g)

        # Step 3 : compute order [r] of [a] modulo [n] with QPE
        self.QPE (a)

        # Measure
        self.measure_all ()
        sim = Aer.get_backend ('qasm_simulator')
        qobj = assemble (self)
        result = sim.run(qobj, shots=1).result ()
        counts = result.get_counts ()

        binary_res = list(counts.keys())[0][:self.t]
        print ("Obtained eigenvalue phi={}".format(binary_res))

        frac = bin2dec (binary_res)
        coefs = continuous_fraction_coefs (frac, 2**self.t)

        for i in range(len(coefs)):
            _, possible_r = partial_sum(coefs[:i+1])

            if possible_r != 0 and pow(a, possible_r, self.n) == 1:
                print ("Found the order of {} mod {} : it's r={}".format(a, self.n, possible_r))

                if possible_r % 2 == 1:
                    print ("But it's not even :/")
                    return None

                elif pow(a, possible_r//2, self.n) == self.n-1:
                    print ("But a^{r/2} is a square root of identity :/")
                    return None

                # Step 4
                p = gcd(pow(a, possible_r//2, self.n) + 1, self.n)
                q = self.n // p
                print ("Bouyah ! Factored p={} and q={}".format(p, q))
                return (p,q)

        print ("Sorry, we didn't find the order :/")



class Shor15 (ShorComputer):
    """
    Factor 15 with Shor's algorithm
    """
    def __init__ (self):
        ShorComputer.__init__ (self, 15)

    def null_state (self):
        self.__init__ ()

    def cu2k (self, a, k, control):
        """
        Hard coded CU^2^k for [a] and [k], for n = 15
        """
        # Only elements which have gcd (a,n) = 1
        if a == 2:
            if k == 0:
                self.cswap (control, 3, 2)
                self.cswap (control, 2, 1)
                self.cswap (control, 1, 0)

            elif k == 1:
                self.cswap (control, 2, 0)
                self.cswap (control, 3, 1)

        elif a == 4:
            if k == 0:
                self.cswap (control, 2, 0)
                self.cswap (control, 3, 1)

        elif a == 7:
            if k == 0:
                self.cswap (control, 1, 0)
                self.cswap (control, 2, 1)
                self.cswap (control, 3, 2)
                for i in range (4):
                    self.cx (control, i)

            if k == 1:
                self.cswap (control, 2, 0)
                self.cswap (control, 3, 1)

        elif a == 8:
            if k == 0:
                self.cswap (control, 1, 0)
                self.cswap (control, 2, 1)
                self.cswap (control, 3, 2)

            elif k == 1:
                self.cswap (control, 2, 0)
                self.cswap (control, 3, 1)

        elif a == 11:
            if k == 0:
                self.cswap (control, 2, 0)
                self.cswap (control, 3, 1)
                for i in range (4):
                    self.cx (control, i)

        elif a == 13:
            if k == 0:
                self.cswap (control, 3, 2)
                self.cswap (control, 2, 1)
                self.cswap (control, 1, 0)
                for i in range (4):
                    self.cx (control, i)

            if k == 1:
                self.cswap (control, 2, 0)
                self.cswap (control, 3, 1)

        elif a == 14:
            if k == 0:
                for i in range (4):
                    self.cx (control, i)


qc = Shor15 ()
qc.run ()
