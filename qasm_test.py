from hhl_functions import *
from qiskit import transpile
from qiskit.qasm2 import dumps
from qmio import QmioRuntimeService



vector = np.array([1]*2)
matrix = tridiag_matrix(2,-1,2)

hhl = HHL(matrix,vector)

# Eliminamos el reset con el que comienza initilize
hhl.qc = hhl.qc.decompose(reps=8)
hhl.qc.data = [instruction for instruction in hhl.qc.data if instruction.operation.name!='reset']

hhl.qc.measure_all()

qc = dumps(hhl.qc)

serv = QmioRuntimeService()

with serv.backend(name="qpu") as backend:
    results = backend.run(circuit=qc,shots=8192)

print(results)

f = open('results.dat','a+')
f.write(str(results))
f.close()