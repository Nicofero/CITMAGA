from mpi4py import MPI
from qulacs import QuantumCircuit, Observable, QuantumState
from qulacs.gate import DenseMatrix
import numpy as np
import time
import re
from math import pi

def parse_qasm3(qasm_code):
    lines = qasm_code.strip().split('\n')
    qubit_dict = {}
    operations = []

    for line in lines:
        if line.startswith("qubit"):
            # Extract the qubit declaration
            match = re.search(r'qubit\[(\d+)\] (\w+);', line)
            if match:
                count, name = int(match.group(1)), match.group(2)
                qubit_dict[name] = count
            else:
                match = re.search(r'qubit (\d+);', line)
                if match:
                    name = match.group(1)
                    qubit_dict[name] = 1
        elif line.startswith("U("):
            # Extract the parameters and target qubit for U gate
            match = re.search(r'U\(([^)]+)\) (\w+)\[(\d+)\];', line)
            if match:
                params = [eval(param.strip()) for param in match.group(1).split(',')]
                qubit = match.group(2)
                target = int(match.group(3))
                operations.append(("U", params, qubit, target))
        elif line.startswith("cx"):
            # Extract the control and target qubits for CX gate
            match = re.search(r'cx (\w+)\[(\d+)\], (\w+)\[(\d+)\];', line)
            if match:
                control_qubit, control_index = match.group(1), int(match.group(2))
                target_qubit, target_index = match.group(3), int(match.group(4))
                operations.append(("CX", control_qubit, control_index, target_qubit, target_index))
    
    return qubit_dict, operations

def qasm3_to_qulacs(qasm_code):
    qubit_dict,operations = parse_qasm3(qasm_code)
    total_qubits = sum(qubit_dict.values())
    circuit = QuantumCircuit(total_qubits)
    
    # Create a mapping from (qubit name, index) to linear index
    qubit_map = {}
    current_index = 0
    for name, count in qubit_dict.items():
        for i in range(count):
            qubit_map[(name, i)] = current_index
            current_index += 1
    
    # Add gates to the circuit
    for op in operations:
        if op[0] == "U":
            theta, phi, lam = op[1]
            qubit_name = op[2]
            target = qubit_map[(qubit_name, op[3])]
            circuit.add_U3_gate(target, theta, phi, lam)
        elif op[0] == "CX":
            control_qubit_name, control_index = op[1], op[2]
            target_qubit_name, target_index = op[3], op[4]
            control = qubit_map[(control_qubit_name, control_index)]
            target = qubit_map[(target_qubit_name, target_index)]
            circuit.add_CNOT_gate(control, target)
    
    return circuit,total_qubits

# Inicialización MPI

mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()
mpisize = mpicomm.Get_size()
globalqubits = int(np.log2(mpisize))

if mpirank==0:
    print("\n\n-------------PRUEBA QULACS-------------\n\n")

with open('qasm_16.qasm','r') as file:
    qasm_code = file.read()

hhl_qul,n_qubits = qasm3_to_qulacs(qasm_code)

start = time.time()
state = QuantumState(n_qubits,use_multi_cpu = True)
hhl_qul.update_quantum_state(state)
state_vector = state.get_vector().real
end= time.time()

num = 2**(n_qubits-1)

if mpirank == 0:
    print(np.round(state_vector,5))
    print("Time with Qulacs: ",end-start)

    print("Solución:",state_vector[num:num+2**4])