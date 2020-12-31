from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

qc = QuantumCircuit(2,2)
backend = Aer.get_backend("qasm_simulator")

qc.h(0)
qc.t(0)
qc.h(0)
qc.h(1)

qc.measure(0,0)
before = execute(qc, backend,shots=1).result().get_counts()
#plot_histogram(counts)

print(before)
qc.draw(output='mpl')
plt.show()

qc.measure(1,1)
after = execute(qc, backend,shots=1).result().get_counts()
#plot_histogram(counts)
#plt.show()

print(after)
