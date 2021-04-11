from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_bloch_vector
from matplotlib import pyplot as plt
from numpy import pi
import numpy as np

# Pretty print style for numpy
np.set_printoptions(precision=3, suppress=True, linewidth=210)

#########################################################################################
###############################           QFT          ##################################
#########################################################################################

def V (k):
    U = QuantumCircuit(2)
    U.cp( pi / (2**k), 0, 1)
    gate = U.to_gate()
    gate.label = "V{}".format(k)
    return gate

def swap (qc, begin_qbit, end_qbit):
    for i in range((end_qbit-begin_qbit-1)//2 + 1):
        qc.swap(begin_qbit+i, end_qbit-i)

def QFT (begin_qbit, end_qbit):
    U = QuantumCircuit(end_qbit-begin_qbit+1)
    for i in range(end_qbit, begin_qbit-1, -1):
        U.h(i)
        for j in range(i):
            U.append(V(i-j), [i,j])
    swap(U, begin_qbit, end_qbit)
    gate = U.to_gate()
    gate.label = "QFT"
    #U.draw(output='mpl')
    #plt.show()
    return gate


#########################################################################################
##############################           Uf           ###################################
#########################################################################################


def controlled_mult7 (qc, begin_output, control):
    """
    Hard coded controlled multiplication by 7 mod 15
    """
    qc.cswap(control, begin_output, begin_output+3)
    qc.cswap(control, begin_output, begin_output+1)
    qc.cswap(control, begin_output+1, begin_output+2)
    for i in range(begin_output, begin_output+4):
        qc.cx(control, i)


def controlled_mult4 (qc, begin_output, control):
    """
    Hard coded controlled multiplication by 4 mod 15
    """
    qc.cswap(control, begin_output, begin_output+2)
    qc.cswap(control, begin_output+1, begin_output+3)

def Uf (begin_input, begin_output):
    """
    Controlled b^x mod N gate, hard coded for b = 7, N = 15, with n0 = 4 (and n=8)
    x is the input
    """
    U = QuantumCircuit(12)
    U.x(begin_output)
    work = 7
    for i in range(8):
        if work == 7:
            controlled_mult7(U, begin_output, begin_input+i)
        elif work == 4:
            controlled_mult4(U, begin_output, begin_input+i)
        work = (work * work) % 15
    gate = U.to_gate()
    gate.label = 'Uf'
    return gate


#########################################################################################
############################           Shor           ###################################
#########################################################################################

def bin_to_dec (m):
    ret = 0
    k = 1
    while m != '':
        ret += k * (int(m[-1])%2)
        m = m[:-1]
        k <<= 1
    return ret

def continuous_fraction_coefs (b, a):
    """
    Continuous fraction coefficients of a/b
    """
    ret = []
    while a%b != 0:
        ret.append(int(a//b))
        a, b = b, a%b
    ret.append(int(a//b))
    return ret


def gcd (a, b):
    while a%b != 0:
        a, b = b, a%b
    return b


def partial_sum (coefs):
    """
    [coefs] is of the form [a0, ..., an], and we compute 1/( a0 + 1/(a1 + ...) ) as p/q, gcd(p,q)=1
    """
    if len(coefs) == 1:
        return (1,coefs[0])
    p,q = partial_sum(coefs[1:])
    den = coefs[0]*q + p
    num = q
    d = gcd(num,den)
    return (int(num//d), int(den//d))


def shor ():
    # Problem constants
    p = 3
    q = 5
    N = p*q
    n0 = 4
    n = 2*n0

    ###############################################################
    ####################    Quantum part  #########################
    ###############################################################

    qc = QuantumCircuit(n+n0, n+n0)

    ### Applying Hadamards ###
    for i in range(n):
        qc.h(i)

    ### Applying Uf ###
    qc.append(Uf(0,n), range(n+n0))

    ### Measuring output register ###
    for j in range(n, n+n0):
        qc.measure(j,j)

    ### Applying QFT on input register ###
    qc.append(QFT(0,n-1), range(n))

    ### Measuring input register ###
    for i in range(n):
        qc.measure(i, i)

    shots = 1024
    counts = execute(qc, Aer.get_backend("qasm_simulator"), shots=shots).result().get_counts()

    ###############################################################
    ####################    Classic part  #########################
    ###############################################################

    # We check how many times a 4 appears in partial sums (which is the order of 7 mod 15)
    good_results = 0

    for y_bin in counts:

        y = bin_to_dec(y_bin[4:])
        d = gcd(y, 2**n)
        approx = (int(y//d), int((2**n)//d)) # y / 2^n

        if approx[0] != 0:

            # computing continuous fraction coefficients
            coefs = continuous_fraction_coefs (approx[0], approx[1])

            # computing continuous fraction partial sums
            for i in range(len(coefs)):
                possible_r = partial_sum(coefs[:i+1])[1]
                if possible_r == 4:
                    good_results += counts[y_bin]
                    break

    print("Average good results : ", good_results/shots)
    qc.draw(output='mpl')
    plt.show()


#########################################################################################
############################           Tests           ##################################
#########################################################################################

def computational_basis_QFT (qc, x, n):
    ### Preparing state ###
    for j in range(n):
        if (x & 1 == 1):
            qc.x(j)
        x >>= 1

    statevector = execute(qc, Aer.get_backend('statevector_simulator')).result().get_statevector()
    plot_bloch_multivector(statevector)

    ### Applying QFT ###
    qc.append(QFT(0,n-1), range(n))

    ### Results ###
    statevector = execute(qc, Aer.get_backend('statevector_simulator')).result().get_statevector()
    plot_bloch_multivector(statevector)
    plt.show()


if __name__ == '__main__':
    shor ()
    #qc = QuantumCircuit(4)
    #computational_basis_QFT(qc, 2, 4)
