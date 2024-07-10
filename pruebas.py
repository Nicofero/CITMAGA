from hhl_functions import *
from qiskit import transpile, QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.quantum_info import Statevector
from qmiotools.integrations.qiskitqmio import QmioBackend, FakeQmio

vector = np.array([1]*4)
matrix = tridiag_matrix(2,-1,4)

hhl = HHL(matrix,vector)

hhl.qc = hhl.qc.decompose(reps=8)
hhl.qc.data = [instruction for instruction in hhl.qc.data if instruction.operation.name!='reset']

hhl.qc.measure_all()

backend = QmioBackend()
backend2 = FakeQmio()

results = hhl.get_counts(backend,shots=8192)

print(results)

prob_ampl = np.sqrt(hhl.prob_from_counts_hhl(results))

print("Estimated amplitudes of the solution:", prob_ampl)

norm = hhl.norm_from_counts(results)

print("Solution: ",norm*prob_ampl)

hist = plot_histogram(results)

plt.savefig('hist.png')