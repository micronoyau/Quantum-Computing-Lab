from qiskit import QuantumCircuit, execute, Aer
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def csqrtx (qc, s, t):
    qc.t(s) # Phase kickback
    qc.h(t)
    qc.t(t)
    qc.cx(s,t)
    qc.tdg(t)
    qc.cx(s,t)
    qc.h(t)

def csqrtxdg (qc, s, t):
    qc.tdg(s) # Phase kickback
    qc.h(t)
    qc.tdg(t)
    qc.cx(s,t)
    qc.t(t)
    qc.cx(s,t)
    qc.h(t)

backend = Aer.get_backend("unitary_simulator")
qc = QuantumCircuit(3)

csqrtx(qc,2,0)
qc.barrier()
qc.cx(2,1)
qc.barrier()
csqrtxdg(qc,1,0)
qc.barrier()
qc.cx(2,1)
qc.barrier()
csqrtx(qc,1,0)
qc.barrier()

qc.draw(output='mpl')

unitary = execute(qc, backend).result().get_unitary()
print(unitary)

plt.show()
