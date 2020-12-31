from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from matplotlib import pyplot as plt
import numpy as np

backend = Aer.get_backend("statevector_simulator")
qc = QuantumCircuit(2)

qc.h(0)
qc.h(1)
qc.z(1)

statevector = execute(qc, backend).result().get_statevector()
with np.printoptions(precision=3, suppress=True):
    print(statevector)
plot_bloch_multivector(statevector)
plot_histogram(execute(qc, backend).result().get_counts())

qc.draw(output='mpl')
plt.show()
