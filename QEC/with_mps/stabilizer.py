from mps import QuantumComputer

class StabilizerCode:

    def __init__ (self, n, k, S, H, Z_bar, X_bar):
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
        self.qc = QuantumComputer (n+1, 32) # 1 ancilla qbit
        self.S = S
        self.H = H
        self.Z_bar = Z_bar
        self.X_bar = X_bar
        self.logical_zero ()

    def operator (self, check_matrix, index):
        """
        Apply n qbit operator defined in [index] row of [check_matrix]
        """
        for j in range (self.n):
            if check_matrix[index,j] == 1:
                self.qc.x (self.n-1-j) # Here n-1-j instead of j because check matrix in reverse order

        for j in range (self.n):
            if check_matrix[index, self.n + j] == 1:
                self.qc.z (self.n-1-j)


    def measure_operator (self, check_matrix, index):
        """
        Measure unitary operator defined by its check matrix [check_matrix] and its row number [index],
        on ancilla qbit [ancilla]
        """
        self.qc.h (self.n)

        for j in range (self.n):
            if check_matrix[index,j] == 1:
                self.qc.cx (self.n, self.n-1-j)

        for j in range (self.n):
            if check_matrix[index, self.n + j] == 1:
                self.qc.cz (self.n, self.n-1-j)

        self.qc.h (self.n)

        meas = self.qc.measure (self.n)
        if meas == 1:
            self.qc.x (self.n)

        return meas


    def correct (self):
        """
        Correct any correctible error.
        """
        beta = []
        for i in range (self.n - self.k):
            beta.append (self.measure_operator (self.S, i))
        print ("Measured syndrome beta = " + str(beta))


    def logical_zero (self):
        """
        Set logical zero on system.
        """
        for i in range (self.n - self.k):
            # Measuring generator i
            meas = self.measure_operator (self.S, i)

            # Stabilize by generator i
            if meas == 1:
                print ("Measured eigenvalue -1 for g" + str(i+1) + ", applying h" + str(i+1))
                # Apply h
                self.operator (self.H, i)

            else:
                print ("Measure eigenvalue +1 for g" + str(i+1) + ", nothing to do")

        """
        for i in range (self.k):
            # Measuring logical Z operator
            meas = self.measure_operator (self.Z_bar, i, self.n-self.k+i)

            # Stabilize by logical Z operator
            if meas == 1:
                # Apply logical X operator
                self.operator (self.X_bar, i)
                # Reset ancilla to |0>
                self.qc.x (self.n - self.k + i)
        """
