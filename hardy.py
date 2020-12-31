from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from math import pi
from matplotlib import pyplot as plt

qc = QuantumCircuit (2)

theta = 2*0.615479
qc.ry(pi/4, 1)
qc.ry(theta, 0)
qc.cx(0,1)
qc.ry(-pi/4, 1)
qc.h(0)

result = execute(qc, Aer.get_backend("statevector_simulator")).result()
#plot_bloch_multivector(result.get_statevector())
plot_histogram(result.get_counts())
qc.draw(output='mpl')

print(result.get_statevector())

plt.show()
