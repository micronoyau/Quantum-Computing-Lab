# Quantum Ciruits with qiskit
from qiskit import QuantumCircuit, Aer, execute

# Visualization tools
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

# Graph tools with networkX
import networkx as nx

# Math
from numpy import pi
from random import random
from scipy.optimize import minimize


def U_gamma_C (gamma, graph):
    n = len(list(graph.nodes()))
    U = QuantumCircuit (n)
    for edge in graph.edges:
        U.cx(edge[0], edge[1])
        U.rz(-gamma, edge[0])
        U.cx(edge[0], edge[1])
    gate = U.to_gate()
    gate.label = "U({:.2f},C)".format(gamma)
    return gate

def U_beta_B (beta, graph):
    n = len(list(graph.nodes()))
    U = QuantumCircuit (n)
    for i in graph.nodes:
        U.h(i)
        U.rz(beta, i)
        U.h(i)
    gate = U.to_gate()
    gate.label = "U({:.2f},B)".format(beta)
    return gate


def setup_ansatz (qc, graph, gamma, beta):
    """
    Creates the state |gamma, beta >, with underlying p size (len(gamma) = len(beta) = p)
    """
    n = len(list(graph.nodes()))
    p = len(gamma)

    for i in range (n):
        qc.h(i)

    for i in range (p):
        qc.append( U_gamma_C (gamma[i], graph), list(range(n)) )
        qc.append( U_beta_B (beta[i], graph), list(range(n)) )


def plot_ansatz (graph, gamma, beta, shots=1024):
    """
    Plots the classical value associated with the quantum state
    in terms of probabilites associated with states
    """
    n = len(list(graph.nodes()))
    qc = QuantumCircuit( n )
    setup_ansatz (qc, graph, gamma, beta)
    qc.measure_all()

    counts = execute(qc, Aer.get_backend("qasm_simulator"), shots=shots).result().get_counts()
    plot_histogram(counts)
    plt.show()


def compute_mean_C (counts, i, j):
    """
    Computes the mean value <psi | Z_i Z_j |psi> with the results of the measures.
    """
    total = sum(counts.values())
    n = len( list(counts.keys())[0] )

    # Probabilities of having ( psi & (2^i + 2^j) ) = ab with a, b in {0,1}
    p_00 = 0
    p_01 = 0
    p_10 = 0
    p_11 = 0

    for key in counts:
        if key[n-1-i] == '0' and key[n-1-j] == '0':
            p_00 += counts[key]
        elif key[n-1-i] == '0' and key[n-1-j] == '1':
            p_01 += counts[key]
        elif key[n-1-i] == '1' and key[n-1-j] == '0':
            p_10 += counts[key]
        else:
            p_11 += counts[key]

    return (p_00 - p_01 - p_10 - p_11) / total


def f (graph, gamma, beta, shots=1024):
    """
    Computes the mean energy value (objective function) <gamma, beta | C | gamma, beta>
    """
    n = len(list(graph.nodes()))

    qc = QuantumCircuit (n)
    setup_ansatz(qc, graph, gamma, beta)
    qc.measure_all()

    # Visualize circuit
    #qc.draw(output='mpl')
    #plt.show()

    counts = execute(qc, Aer.get_backend("qasm_simulator"), shots=shots).result().get_counts()

    res = 0
    for edge in graph.edges:
        mean = compute_mean_C (counts, edge[0], edge[1])
        res += 1 - mean
    res /= 2

    return res


if __name__ == '__main__':
    # Testing on some graph
    G = nx.Graph()
    G.add_nodes_from(list(range(0,5)));
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(1,2)
    G.add_edge(1,4)
    G.add_edge(2,3)
    G.add_edge(4,3)

    def opt (angles):
        # Remember we want to optimize negative f to find lowest eigenvalue
        return -f(G, angles[:len(angles)//2], angles[len(angles)//2:])


    # Plotting graph
    #nx.draw(G)
    #plt.show()

    p = 3
    gamma = [ random() * 2 * pi for i in range(p) ]
    beta = [ random() * 2 * pi for i in range(p) ]

    #print(f(G, gamma, beta))
    res = minimize (opt, gamma + beta, method='COBYLA', options={'maxiter':50, 'rhobeg':0.001})

    lambda_max = -res['fun']
    opt_angles = res['x']

    print( "Found maximum eigenvalue : ", lambda_max )

    plot_ansatz (G, opt_angles[:len(opt_angles)//2], opt_angles[len(opt_angles)//2:])
