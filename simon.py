########################### Importations ###########################

import matplotlib.pyplot as plt

from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, execute

from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import simon_oracle


########################### Implementation ###########################

b = '101'
n = len(b)

circuit = QuantumCircuit(n*2, n)

circuit.h(range(n))
circuit.barrier()

circuit += simon_oracle(b[::-1])

circuit.barrier()
circuit.h(range(n))

circuit.measure(range(n), range(n))

print(circuit)
circuit.draw(output='mpl')

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
results = execute(circuit, backend=backend, shots=shots).result()
counts = results.get_counts()

plot_histogram(counts)

print(counts)

def bdotz(b, z):
    accum = 0
    for i in range(len(b)):
        accum += int(b[i]) * int(z[i])
    return (accum % 2)

for z in counts:
    print( '{}.{} = {} (mod 2)'.format(b, z, bdotz(b,z)) )

plt.show()