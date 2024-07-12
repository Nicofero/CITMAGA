from hhl_functions import *
from qiskit import transpile, QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.quantum_info import Statevector
from qmiotools.integrations.qiskitqmio import QmioBackend, FakeQmio

# Only change this
nb=1

NB = 2**nb
vector = np.array([1]*NB)
matrix = tridiag_matrix(2,-1,NB)

hhl = HHL(matrix,vector,flag=False)

hhl.qc = hhl.qc.decompose(reps=8)
hhl.qc.data = [instruction for instruction in hhl.qc.data if instruction.operation.name!='reset']

hhl.qc.measure_all()

backend = QmioBackend()
backend2 = FakeQmio()

results = hhl.get_counts(backend2)

prob_ampl = np.sqrt(hhl.prob_from_counts_hhl(results))
num = int(len(prob_ampl)/2)
sol = prob_ampl[num:num+NB]

print("Estimated amplitudes of the solution:", sol/np.linalg.norm(sol))

norm = hhl.norm_from_counts(results)

print("Solution: ",norm*sol)

hist = plot_histogram(results)

plt.savefig('hist.png')