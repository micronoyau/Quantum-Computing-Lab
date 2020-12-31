from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt
import numpy as np

backend = Aer.get_backend("statevector_simulator")
qc = QuantumCircuit (2)

qc.h(1)
qc.cx(1,0)
qc.x(0)

vector = execute(qc, backend).result().get_statevector()
counts = execute(qc, backend).result().get_counts()
unitary = execute(qc, Aer.get_backend("unitary_simulator")).result().get_unitary()

#qc.draw(output='mpl')
#plot_histogram(counts)
print("result vector : ",  vector)
print("unitary : ")
with np.printoptions(precision=3, suppress=True):
    print(unitary)

#plt.show()
