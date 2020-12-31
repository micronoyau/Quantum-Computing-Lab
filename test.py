from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from matplotlib import pyplot as plt

backend = Aer.get_backend("statevector_simulator")

qc = QuantumCircuit (2)

qc.sdg(0)
qc.cx(1,0)
qc.s(0)

"""
result = execute (qc, backend).result()
vector = result.get_statevector()
print(vector)
plot_bloch_multivector(vector)
plt.show()
"""
qc.draw(output='mpl')
plt.show()
