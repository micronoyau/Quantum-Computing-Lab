from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

backend = Aer.get_backend("statevector_simulator")
qc = QuantumCircuit(2)
qc.h(0) # State : |0+>
qc.cx(0,1) # C01 |0+>
vector = execute(qc, backend).result().get_statevector()
counts = execute(qc, backend).result().get_counts()

qc.draw(output='mpl')
plot_histogram(counts)
print(vector)
plt.show()
