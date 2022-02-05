from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from matplotlib import pyplot as plt
import networkx as nx

class SurfaceCode:
    """
    Basic surface code to encode a single logical qbit
    """

    def __init__ (self, d):
        """
        Create a [d^2+(d-1)^2,1] surface code
        """
        self.d = d
        self.data = QuantumRegister (d*d + (d-1)*(d-1), 'data') # Data qbits
        self.gz = QuantumRegister (d*(d-1), 'gz') # Z measurement qbits
        self.gx = QuantumRegister (d*(d-1), 'gx') # X measurement qbits
        self.sz = list () # list of Z syndrome in classical bits : 0 = +1 eigenvalue, 1 = -1 eigenvalue
        self.sx = list () # list of X syndrome in classical bits : 0 = +1 eigenvalue, 1 = -1 eigenvalue
        self.qc = QuantumCircuit (self.data, self.gz, self.gx)
        self.rounds = 0 # Number of cycles

    def prepare_cycle (self):
        """
        Create new classical registers to hold the value of further stabilizer measurements
        """
        self.sz.append (ClassicalRegister (self.d*(self.d-1), 'sz' + str(len(self.sz))))
        self.sx.append (ClassicalRegister (self.d*(self.d-1), 'sx' + str(len(self.sx))))
        self.qc.add_register (self.sz[-1])
        self.qc.add_register (self.sx[-1])

    def step1 (self):
        # Do nothing for Z generators
        self.qc.id (self.gz)
        # Initialize X generators
        self.qc.reset (self.gx)
        self.qc.barrier ()

    def step2 (self):
        # Initialize Z generators
        self.qc.reset (self.gz)
        # Hadamard X generators
        self.qc.h (self.gx)
        self.qc.barrier ()

    def step3 (self):
        # Upper measurement for both Z and X
        for i in range ((self.d-1)**2):
            row = i // (self.d-1)
            data_offset = self.d * (row+1) + (self.d-1) * row
            gz_offset = (self.d-1) * (row+1)
            row_offset = i % (self.d-1)
            self.qc.cx (self.data[data_offset + row_offset], self.gz[gz_offset + row_offset])

        for i in range (self.d*(self.d-1)):
            row = i // self.d
            data_offset = (2*self.d-1) * row
            gx_offset = self.d * row
            row_offset = i % self.d
            self.qc.cx (self.gx[gx_offset + row_offset], self.data[data_offset + row_offset])
        self.qc.barrier ()

    def step4 (self):
        # Right measurement for both Z and X
        for i in range (self.d*(self.d-1)):
            row = i // (self.d-1)
            data_offset = 1 + (2*self.d-1) * row
            gz_offset = (self.d-1) * row
            row_offset = i % (self.d-1)
            self.qc.cx (self.data[data_offset + row_offset], self.gz[gz_offset + row_offset])

        for i in range ((self.d-1)**2):
            row = i // (self.d-1)
            data_offset = self.d + (2*self.d-1) * row
            gx_offset = self.d * row
            row_offset = i % (self.d-1)
            self.qc.cx (self.gx[gx_offset + row_offset], self.data[data_offset + row_offset])
        self.qc.barrier ()

    def step5 (self):
        # Left measurement for both Z and X
        for i in range (self.d*(self.d-1)):
            row = i // (self.d-1)
            data_offset = row * (2*self.d-1)
            gz_offset = (self.d-1) * row
            row_offset = i % (self.d-1)
            self.qc.cx (self.data[data_offset + row_offset], self.gz[gz_offset + row_offset])

        for i in range ((self.d-1)**2):
            row = i // (self.d-1)
            data_offset = self.d + (2*self.d-1) * row
            gx_offset = 1 + row * self.d
            row_offset = i % (self.d-1)
            self.qc.cx (self.gx[gx_offset + row_offset], self.data[data_offset + row_offset])
        self.qc.barrier ()

    def step6 (self):
        # Downard measurement for both Z and X
        for i in range ((self.d-1)**2):
            row = i // (self.d-1)
            data_offset = self.d * (row+1) + (self.d-1) * row
            gz_offset = (self.d-1) * row
            row_offset = i % (self.d-1)
            self.qc.cx (self.data[data_offset + row_offset], self.gz[gz_offset + row_offset])

        for i in range (self.d*(self.d-1)):
            row = i // self.d
            data_offset = (2*self.d-1) * (row+1)
            gx_offset = self.d * row
            row_offset = i % self.d
            self.qc.cx (self.gx[gx_offset + row_offset], self.data[data_offset + row_offset])
        self.qc.barrier ()

    def step7 (self):
        self.qc.measure (self.gz, self.sz[-1])
        self.qc.h (self.gx)
        self.qc.barrier ()

    def step8 (self):
        self.qc.id (self.gz)
        self.qc.measure (self.gx, self.sx[-1])
        self.qc.barrier ()

    def _extract_de_graph (self, primal_meas, dual_meas):
        primal = nx.grid_graph (dim=( range(self.rounds-1), # z coord
                                      range(self.d), # y coord
                                      range(self.d-1) )) # x coord
        dual = nx.grid_graph (dim=( range(self.rounds-1), # z coord
                                    range(self.d-1), # y coord
                                    range(self.d) )) # x coord

        primal_de_graph = nx.Graph () # Primal detection events weighted graph
        dual_de_graph = nx.Graph () # Dual detection events weighted graph

        for i in range (len(primal_meas)-1):
            # Detecting the generators that differ
            primal_diff = int(primal_meas[i], base=2) ^ int(primal_meas[i+1], base=2)
            gen_location = 0
            while primal_diff != 0:
                # A generator measurement that differs from one measure to another
                if primal_diff & 1 == 1:
                    x = gen_location % (self.d - 1)
                    y = gen_location // (self.d - 1)
                    z = i
                    # Add it to the weighted graph
                    primal_de_graph.add_node ( (x, y, z) )
                primal_diff >>= 1
                gen_location += 1

        for i in range (len(dual_meas)-1):
            dual_diff = int(dual_meas[i], base=2) ^ int(dual_meas[i+1], base=2)
            gen_location = 0
            while dual_diff != 0:
                if dual_diff & 1 == 1:
                    x = gen_location % self.d
                    y = gen_location // self.d
                    z = i
                    dual_de_graph.add_node ( (x, y, z) )
                dual_diff >>= 1
                gen_location += 1

        # Complete graphs
        primal_de_graph = nx.complete_graph (primal_de_graph.nodes)
        dual_de_graph = nx.complete_graph (dual_de_graph.nodes)
        nx.set_edge_attributes (primal_de_graph, 1, 'weight')
        nx.set_edge_attributes (dual_de_graph, 1, 'weight')

        # Compute the shortest paths and report them in weights
        for edge in primal_de_graph.edges(data=True):
            weight = nx.shortest_path_length (primal, source=edge[0], target=edge[1])
            edge[2]['weight'] = weight

        for edge in dual_de_graph.edges(data=True):
            weight = nx.shortest_path_length (dual, source=edge[0], target=edge[1])
            edge[2]['weight'] = weight

        return (primal_de_graph, dual_de_graph)


    def post_analysis (self):
        """
        In a real world quantum computer, we could instead perform this step after each cycle,
        to correct on the fly. Here, qiskit does not allow us to do this, so we have to wait
        the end of all cycles to get the correct result.
        """
        print ("Performing post-analysis ...")

        # Actual simulation results
        counts = execute (s.qc, Aer.get_backend('qasm_simulator'), shots=1).result().get_counts()
        print(counts)
        counts = list(counts.keys())[0]
        meas = counts.split(' ')[::-1]
        primal_meas = meas[::2]
        dual_meas = meas[1::2]

        print ("Simulation results :")
        print ("Primal :")
        print(primal_meas)
        print ("Dual :")
        print(dual_meas)

        print ("Computing detection event graphs ...")
        primal_de_graph, dual_de_graph = self._extract_de_graph (primal_meas, dual_meas)

        print ("Got the following weights :")
        print ("Primal :")
        print(primal_de_graph.edges(data=True))
        print ("Dual :")
        print(dual_de_graph.edges(data=True))

        nx.draw (primal_de_graph, with_labels=True, node_color="red")
        nx.draw (dual_de_graph, with_labels=True, node_color="green")

        plt.show ()

        print ("Computing matchings ...")

        primal_matching = nx.min_weight_matching (primal_de_graph)
        dual_matching = nx.min_weight_matching (dual_de_graph)

        print ("Primal :")
        print (primal_matching)
        print ("Dual :")
        print (dual_matching)

    def _apply_cycle_errors (self, errors, cycle):
        if cycle in errors:
            for (error_t, qbit) in errors[cycle]:
                if error_t == 'X':
                    self.qc.x (qbit)
                elif error_t == 'Y':
                    self.qc.y (qbit)
                elif error_t == 'Z':
                    self.qc.z (qbit)


    def cycle (self, errors={}):
        """
        [errors] is a dictionnary of errors that happen during a cycle, where each element ok key [i]
        is a tuple (error_type, qbit) that represents what type of error occurs on which qbit at
        the end of cycle  [i].
        """
        self.prepare_cycle ()
        self.step1 ()
        self._apply_cycle_errors (errors, 1)
        self.step2 ()
        self._apply_cycle_errors (errors, 2)
        self.step3 ()
        self._apply_cycle_errors (errors, 3)
        self.step4 ()
        self._apply_cycle_errors (errors, 4)
        self.step5 ()
        self._apply_cycle_errors (errors, 5)
        self.step6 ()
        self._apply_cycle_errors (errors, 6)
        self.step7 ()
        self._apply_cycle_errors (errors, 7)
        self.step8 ()
        self.rounds += 1

s = SurfaceCode (3)
s.cycle ()
s.cycle (errors={5:[('X', 3),]})
s.cycle (errors={3:[('X', 6),]})
s.cycle ()
s.post_analysis ()

#counts = execute (s.qc, Aer.get_backend('qasm_simulator'), shots=10).result().get_counts()
#print(counts)
#s.qc.draw (output='mpl', style='bw', justify='left', fold=-1)
#plt.show()
