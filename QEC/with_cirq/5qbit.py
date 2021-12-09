from stabilizer import StabilizerCode
import cirq
import pickle
import numpy as np

S = np.array ([[1,0,0,1,0, 0,1,1,0,0],
               [0,1,0,0,1, 0,0,1,1,0],
               [1,0,1,0,0, 0,0,0,1,1],
               [0,1,0,1,0, 1,0,0,0,1]])

H = np.array ([[0,1,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,1],
               [0,0,0,0,0, 0,0,1,0,0],
               [1,0,0,0,0, 0,0,0,0,0]])

X_bar = np.array([[0,0,0,0,1, 1,0,0,1,0],])

Z_bar = np.array([[0,0,0,0,0, 1,1,1,1,1],])

sc = StabilizerCode (5, 1, S, H, Z_bar, X_bar, debug=False)

sc.qc.append (cirq.X(sc.qreg[0]))

#sc.operator (X_bar, 0)

sc.check_generators ()
sc.measure_logical ()
#print(sc.qc)
