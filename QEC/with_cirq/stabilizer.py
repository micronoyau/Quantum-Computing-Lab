import cirq
from cirq.circuits import InsertStrategy

class StabilizerCode:

    def __init__ (self, n, k, S, H, Z_bar, X_bar, debug=False):
        """
        Creates a [n,k] code stabilized by S = <g1, ..., g{n-k}>, given by a numpy array
        which represents the check matrix.
        H is the check matrix representing some hi st hi anticommutes with gi but commutes
        with everyone else.
        Z_bar are the chosen logical Z operators.
        X_bar are the chosen logical X operators.
        The gi are independent and commutative.
        """
        self.n = n
        self.k = k

        self.qreg = cirq.LineQubit.range (n)
        self.ancillas = cirq.LineQubit.range (n, 2*n) # n-k ancillas to measure generators, k ancillas to measure logical Z
        self.final = cirq.LineQubit.range (2*n, 3*n) # Check afterwards (other ancillas)
        self.qc = cirq.Circuit ()

        self.S = S
        self.H = H
        self.Z_bar = Z_bar
        self.X_bar = X_bar

        self.logical_zero (debug=debug)


    def operator (self, check_matrix, index):
        """
        Apply n qbit operator defined in [index] row of [check_matrix]
        """
        for j in range (self.n):
            if check_matrix[index,j] == 1:
                self.qc.append (cirq.X (self.qreg[self.n-1-j]), strategy=InsertStrategy.NEW) # Here n-1-j instead of j because check matrix in reverse order

        for j in range (self.n):
            if check_matrix[index, self.n + j] == 1:
                self.qc.append (cirq.Z (self.qreg[self.n-1-j]), strategy=InsertStrategy.NEW)


    def controlled_operator (self, check_matrix, index, ancilla):
        """
        Apply n qbit operator definied in [index] row of check_matrix, controlled by ancilla qbit [ancilla]
        """
        for j in range (self.n):
            if check_matrix[index,j] == 1:
                self.qc.append (cirq.CNOT (ancilla, self.qreg[self.n-1-j]), strategy=InsertStrategy.NEW)

        for j in range (self.n):
            if check_matrix[index, self.n + j] == 1:
                self.qc.append (cirq.CZ (ancilla, self.qreg[self.n-1-j]), strategy=InsertStrategy.NEW)


    def measure_observable (self, check_matrix, index, ancilla):
        """
        Measure unitary observable defined by its check matrix [check_matrix] and its row number [index],
        on ancilla qbit [ancilla]
        If ancilla qbit yields |0> after measurement, it means the observable is measured as +1, and if |1>, we measure -1.
        """
        self.qc.append (cirq.H (ancilla), strategy=InsertStrategy.NEW)
        self.controlled_operator (check_matrix, index, ancilla)
        self.qc.append (cirq.H (ancilla), strategy=InsertStrategy.NEW)
        self.qc.append (cirq.measure (ancilla), strategy=InsertStrategy.NEW)


    def correct (self, debug=False):
        """
        Correct any correctible error.
        """
        for i in range (self.n - self.k):
            # Measure gi
            self.measure_observable (self.S, i, self.ancillas[i])
            # Stabilize by gi
            self.controlled_operator (self.H, i, self.ancillas[i])

        if debug:
            sim = cirq.Simulator ()
            shot = sim.run (self.qc)
            beta = [shot.measurements[str(i)][0,0] for i in range (self.n, 2*self.n-self.k)]
            print ("Measured syndrome beta = " + str(beta))


    def check_generators (self):
        """
        Check generator values
        """
        for i in range (self.n - self.k):
            self.measure_observable (self.S, i, self.final[i])

        sim = cirq.Simulator ()
        shot = sim.run (self.qc)
        beta = [shot.measurements[str(i)][0,0] for i in range (2*self.n, 3*self.n-self.k)]
        print ("Verification : measured syndrome beta = " + str(beta))


    def measure_logical (self):
        """
        Measure logical value
        """
        for i in range (self.k):
            self.measure_observable (self.Z_bar, i, self.final[self.n-self.k+i])

        sim = cirq.Simulator ()
        shot = sim.run (self.qc)
        meas = [shot.measurements[str(i)][0,0] for i in range (3*self.n-self.k, 3*self.n)]
        print ("Measured logical " + ''.join(map(str, meas)))


    def logical_zero (self, debug=False):
        """
        Set logical zero on system.
        """
        self.correct (debug=debug)

        for i in range (self.k):
            # Measure \bar{Z}_i
            self.measure_observable (self.Z_bar, i, self.ancillas[self.n-self.k+i])
            # Stabilize by \bar{Z}_i
            self.controlled_operator (self.X_bar, i, self.ancillas[self.n-self.k+i])
