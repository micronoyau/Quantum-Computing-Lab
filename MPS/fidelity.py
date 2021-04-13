from mps import QuantumComputer
from qiskit import QuantumCircuit
import numpy as np

"""
Measurement of the fidelity of MPS quantum circuit
"""

qc_mps = QuantumComputer (10, 64)
qc_qiskit = QuantumCircuit (10)

def gen_rand_circuit (N, D, khi):
    """
    Generates a random circuit of depth [D] with [N] qbits,
    according to paper, and with entanglement degree [khi]
    """

    qc = QuantumComputer (N, khi)

    for d in range(D):
        # First apply 1 qbit gates (rotations) of angle theta around m(alpha, phi)
        for qbit in range (N):
            alpha = 2 * np.pi * np.random.random()
            phi = 2 * np.pi * np.random.random()
            theta = 2 * np.pi * np.random.random()
            qc.rot (alpha, phi, theta, qbit)

        # Then apply 2 qbit gates
        if D%2 == 0:
            for qbit in range (N//2):
                qc.cz (2 * qbit, 2 * qbit + 1)
        else:
            for qbit in range ((N-1)//2):
                qc.cz (2 * qbit + 1, 2 * qbit + 2)

    return qc

qc = gen_rand_circuit (5, 1, 20)

print(qc)
