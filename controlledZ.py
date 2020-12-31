from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from matplotlib import pyplot as plt

qc = QuantumCircuit (2)
qc.x(0)
qc.h(1)
qc.h(0)
qc.cx(1,0)
qc.h(0)

result = execute(qc, Aer.get_backend("statevector_simulator")).result()

plot_bloch_multivector(result.get_statevector())
#qc.draw(output='mpl')
plt.show()
