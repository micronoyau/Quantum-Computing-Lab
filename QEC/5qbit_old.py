from mps import QuantumComputer
import pickle

def logical_zero ():
    qc = QuantumComputer (10, 32)

    # Measuring g1 = XZZXI
    qc.h (5)
    qc.cx (5, 1)
    qc.cz (5, 2)
    qc.cz (5, 3)
    qc.cx (5, 4)
    qc.h (5)

    # Applying h1 = IXIII if in the -1 eigenspace (stabilized by -g1)
    if (qc.measure (5) == 1):
        qc.x (3)

    # Measuring g2 = IXZZX
    qc.h (6)
    qc.cx (6, 0)
    qc.cz (6, 1)
    qc.cz (6, 2)
    qc.cx (6, 3)
    qc.h (6)

    # Applying h2 = IIIIZ if in the -1 eigenspace (stabilized by -g2)
    if (qc.measure (6) == 1):
        qc.z (0)

    # Measuring g3 = XIXZZ
    qc.h (7)
    qc.cz (7, 0)
    qc.cz (7, 1)
    qc.cx (7, 2)
    qc.cx (7, 4)
    qc.h (7)

    # Applying h3 = IIZII if in the -1 eigenspace (stabilized by -g3)
    if (qc.measure (7) == 1):
        qc.z (2)

    # Measuring g4 = ZXIXZ
    qc.h (8)
    qc.cz (8, 0)
    qc.cx (8, 1)
    qc.cx (8, 3)
    qc.cz (8, 4)
    qc.h (8)

    # Applying h4 = XIIII if in the -1 eigenspace (stabilized by -g4)
    if (qc.measure (8) == 1):
        qc.x (4)

    # Stabilizing by \bar{Z} = ZZZZZ the same way

    qc.h (9)
    qc.cz (9,0)
    qc.cz (9,1)
    qc.cz (9,2)
    qc.cz (9,3)
    qc.cz (9,4)
    qc.h (9)

    if (qc.measure (9) == 1):
        # \bar{X} = ZIIZX
        qc.z (4)
        qc.z (1)
        qc.x (0)

    return qc

#with open ("logical_zero.pickle", "wb") as f:
#    pickle.dump (logical_zero(), f)

with open ("logical_zero.pickle", "rb") as f:
    qc = pickle.load (f)

    qc.x (3)

    if qc.measure (7) == 1:
        qc.x (7)

    qc.h (7)
    qc.cz (7, 0)
    qc.cz (7, 1)
    qc.cx (7, 2)
    qc.cx (7, 4)
    qc.h (7)

    print (qc.measure (7))
