from hhl_functions import *
from qmiotools.integrations.qiskitqmio import QmioBackend

vector = np.array([1]*2)
matrix = tridiag_matrix(2,-1,2)
hhl = HHL(matrix,vector)
size = hhl.nb+hhl.nl+hhl.nf+hhl.na
qc = QuantumCircuit(2*size)
qc.append(hhl.qc,range(size))
qc.append(hhl.qc,range(size,2*size))

st = Statevector(qc).data.real
print(len(st))
print(st)
backend = QmioBackend()

qc = prepare_circ(qc)
qc.measure_all()
qc_meas = transpile(qc,backend,optimization_level=2)

job = backend.run(qc_meas)
job_result = job.result()

results=job_result.get_counts()

all_outcomes = [''.join(outcome) for outcome in product('01', repeat=8)]

shots = sum(results.values())

prob = []
for elem in all_outcomes:
    if elem in results:
        prob.append(results[elem]/shots)
    else:
        prob.append(0)
prob = np.array(prob)
prob_ampl = np.sqrt(prob)

sol = prob_ampl[136:138]
sol2 = prob_ampl[152:154]
print("Estimated amplitudes of the solution 1:", sol/np.linalg.norm(sol))
print("Estimated amplitudes of the solution 2:", sol2/np.linalg.norm(sol2))

hist = plot_histogram(results)

plt.savefig('hist_paralel.png')