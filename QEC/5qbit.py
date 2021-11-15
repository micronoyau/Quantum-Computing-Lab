from stabilizer import StabilizerCode
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

sc = StabilizerCode (5, 1, S, H, Z_bar, X_bar)

with open ("5qbit.pickle", "wb") as f:
    pickle.dump (sc, f)

with open ("5qbit.pickle", "rb") as f:
    sc = pickle.load (f)
    print ("Before error :")
    sc.correct ()

    sc.qc.x (3)
    print ("After error :")
    sc.correct ()
