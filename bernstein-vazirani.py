########################### Importations ###########################

import matplotlib.pyplot as plt
import numpy as np

from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

from qiskit.visualization import plot_histogram

########################### Helpers ###########################

def apply_gate(circuit, n, gate):
    if gate == 'H':
        for i in range(n):
            circuit.h(i)
    elif gate == 'M':
        for i in range(n):
            circuit.measure(i, i)

def apply(circuit, n, callback):
    for i in range(n):
        callback(circuit, i)

########################### Implementation ###########################

n = 4
s = '1011'
U_f = lambda c, x: c.i(x) if s[x] == '0' else c.cx(x, n)


# Circuit
circuit = QuantumCircuit(n+1,n)

circuit.h(range(n+1))
circuit.z(n)

circuit.barrier()
s = s[::-1]
apply(circuit, n, U_f)
circuit.barrier()

circuit.h(range(n))
circuit.measure(range(n),range(n))


print(circuit)
circuit.draw(output='mpl')

# Simulation

backend = BasicAer.get_backend('qasm_simulator')
answer = execute(circuit, backend=backend, shots=8).result().get_counts()

print(answer)

plot_histogram(answer)
plt.show()