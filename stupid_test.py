from hhl_functions import *
from qiskit.qasm3 import dumps

print("\n\n-------------PRUEBA CIRCUITO DIMESION 32-------------\n\n")

vector = np.array([1]*32)
matrix = tridiag_matrix(2,-1,32)

hhl = HHL(matrix,vector)

hhl.qc = prepare_circ(hhl.qc)

result = dumps(hhl.qc)

with open('qasm_32.qasm','a+') as file:
    file.write(result)

print(result)