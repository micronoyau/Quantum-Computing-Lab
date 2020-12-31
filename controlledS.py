from qiskit import QuantumCircuit, execute, Aer
from matplotlib import pyplot as plt

qc = QuantumCircuit (2)
qc.t(0)
qc.t(1)
qc.cx(1,0)
qc.tdg(0)
qc.cx(1,0)

unitary = execute(qc, Aer.get_backend("unitary_simulator")).result().get_unitary()
print(unitary)

qc.draw(output='mpl')
plt.show()
