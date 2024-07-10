from hhl_functions import *
from qiskit import transpile, QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.quantum_info import Statevector
from qmiotools.integrations.qiskitqmio import QmioBackend, FakeQmio
import time

def init_and_qpe(vector,matrix,tolerance=10e-3):
    # Define the tolerances of the circuit
    epsilon_a = tolerance/6
    epsilon_r = tolerance/3
    epsilon_s = tolerance/3
    
    # We need an np.array to write the values to the register
    if isinstance(vector,(list,np.ndarray)):
        if isinstance(vector,list):
            vector = np.array(vector)    
        # We define the number of needed qubits and insert the vector to the register
        nb = int(np.log2(len(vector)))
        vector_circuit = QuantumCircuit(nb)
        # vector_circuit.initialize(vector / np.linalg.norm(vector), list(range(nb)), None)
        for i in range(nb):
            vector_circuit.h(i)
    else:
        raise ValueError(f"Invalid type for vector: {type(vector)}.")
    
    # Define flag, if 1, correct solution, if 0, incorrect
    nf = 1
    
    # Input the matrix A for the QPE
    if isinstance(matrix, (list, np.ndarray)):
        if isinstance(matrix, list):
            matrix = np.array(matrix)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        
        if np.log2(matrix.shape[0]) % 1 != 0:
            raise ValueError("Input matrix dimension must be 2^n!")
        
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Input matrix must be hermitian!")
        
        if matrix.shape[0] != 2 ** vector_circuit.num_qubits:
            raise ValueError(
                "Input vector dimension does not match input "
                "matrix dimension! Vector dimension: "
                + str(vector_circuit.num_qubits)
                + ". Matrix dimension: "
                + str(matrix.shape[0])
            )
        # We default to a TridiagonalToeplitz matrix, but in a general case we would use a more general library
        # Also, we want the evolution_time to be 2pi/\landa_{max}, but we update it after, when we have the eigenvalues of the matrix
        matrix_circuit = NumPyMatrix(matrix,evolution_time=2 * np.pi, tolerance=epsilon_a)
        
        # General case
        # matrix_circuit = NumPyMatrix(matrix, evolution_time=2 * np.pi)
        
    else:
        raise ValueError(f"Invalid type for matrix: {type(matrix)}.")
    
    # Define condition and eigenvalue bounds    
    if (hasattr(matrix_circuit, "condition_bounds")):
        kappa = matrix_circuit.condition_bounds()[1]
    else:
        kappa = 1
    # Using kappa, the condition bound,  we define nl, the number of qubits needed to represent the eigenvalues
    nl = max(nb+1,int(np.log2(kappa))+1)
    # Construction of the circuit
    
    # Initialise the quantum registers
    qb = QuantumRegister(nb,name="b")  # right hand side and solution
    ql = QuantumRegister(nl,name="0")  # eigenvalue evaluation qubits
    qf = QuantumRegister(nf,name="flag")  # flag qubits
    qc = QuantumCircuit(qb, ql, qf)

    # State preparation
    qc.append(vector_circuit, qb[:])
    qc.barrier(label="\pi_1")
    # QPE
    phase_estimation = PhaseEstimation(nl, matrix_circuit)
    qc.append(phase_estimation, ql[:] + qb[:])
    return qc,nb,nl,matrix_circuit
##########################################################################

def eig_inverse(state,nb,nl,matrix_circuit):
    
    # Define condition and eigenvalue bounds    
    if (hasattr(matrix_circuit, "condition_bounds")):
        kappa = matrix_circuit.condition_bounds()[1]
    else:
        kappa = 1
    # Using kappa, the condition bound,  we define nl, the number of qubits needed to represent the eigenvalues
    nl = max(nb+1,int(np.log2(kappa))+1)
    
    # Define eigenvalues
    if hasattr(matrix_circuit, "eigs_bounds"):
        lambda_min, lambda_max = matrix_circuit.eigs_bounds()
        
        # Constant so that the minimum eigenvalue is represented exactly, since it contributes
        # the most to the solution of the system
        delta = get_delta(nl, lambda_min, lambda_max)
        # Update evolution time
        matrix_circuit.evolution_time = 2 * np.pi * delta / lambda_min
        # Update the scaling of the solution
        scaling = lambda_min
    else:
        delta = 1 / (2 ** nl)
        print("The solution will be calculated up to a scaling factor.")
    
        
    # Define the reciprocal circuit
    
    # # Using an exact reciprocal circuit
    
    reciprocal_circuit = ExactReciprocal(nl, delta)
    # Update number of ancilla qubits
    na = matrix_circuit.num_ancillas
    
    # Construction of the circuit
    
    # Initialise the quantum registers
    qb = QuantumRegister(nb,name="b")  # right hand side and solution
    ql = QuantumRegister(nl,name="0")  # eigenvalue evaluation qubits
    if na > 0:
        qa = AncillaRegister(na,name="anc")  # ancilla qubits
    qf = QuantumRegister(1,name="flag")  # flag qubits

    if na > 0:
        qc = QuantumCircuit(qb, ql, qa, qf)
    else:
        qc = QuantumCircuit(qb, ql, qf)

    # State preparation
    qc.initialize(state)
    # Conditioned rotation
    qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
    
    return qc,scaling

##########################################################################

def inverse_qpe(state,nb,nl,matrix_circuit):
       
    # Initialise the quantum registers
    qb = QuantumRegister(nb,name="b")  # right hand side and solution
    ql = QuantumRegister(nl,name="0")  # eigenvalue evaluation qubits
    qf = QuantumRegister(1,name="flag")  # flag qubits
    qc = QuantumCircuit(qb, ql, qf)

    # State preparation
    qc.initialize(state)
    # QPE inverse
    phase_estimation = PhaseEstimation(nl, matrix_circuit)
    qc.append(phase_estimation.inverse(), ql[:] + qb[:])

    return qc

def ampl_from_sim(qc,backend,shots=8192,reps=1):
    qc.remove_final_measurements()
    qc.measure_all()
    qc = transpile(qc,backend,optimization_level=2)
    
    nb = qc.qregs[0].size
    nl = qc.qregs[1].size

    for _ in range(reps):
        job = backend.run(qc,shots=shots)
        job_result = job.result()
        res=job_result.get_counts()
        counts = {k: counts.get(k, 0) + res.get(k, 0) for k in set(res) | set(counts)}

    all_outcomes = [''.join(outcome) for outcome in product('01', repeat=nb+nl+1)]

    prob = []
    for elem in all_outcomes:
        if elem in counts:
            prob.append(counts[elem]/shots)
        else:
            prob.append(0)
    prob = np.array(prob)
    prob = np.sqrt(prob)
    return prob

b_qmio = QmioBackend()
b_fake = FakeQmio()
shots = 8192

# Initialization
vector = np.array([1]*4)
matrix = tridiag_matrix(2,-1,4)

start = time.time()
# First circuit (Statevector)
qc1,nb,nl,mt_circ=init_and_qpe(vector,matrix)
state = Statevector(qc1)

# Second circuit (Qmio)
qc2,_ = eig_inverse(state,nb,nl,mt_circ)
qc2 = prepare_circ(qc2)

state2 = ampl_from_sim(qc2,b_qmio)

# Third circuit (Statevector)
qc3 = inverse_qpe(state2,nb,nl,mt_circ)
state_f = Statevector(qc3)
num = int(len(state_f)/2)
sol = state_f.data.real[num:num+2**nb]
end = time.time()

print(f'Solucion con 1 y 3 con statevectors: {sol/np.linalg.norm(sol)}')
print('Tiempo:',end-start)

start = time.time()
# First circuit (FakeQmio)
qc1 = prepare_circ(qc1)
state = ampl_from_sim(qc1,b_fake)

# Second circuit (Qmio)
state2 = ampl_from_sim(qc2,b_qmio)

# Third circuit (FakeQmio)
qc3 = prepare_circ(qc3)
state_f = ampl_from_sim(qc3,b_fake)
sol = state_f.data.real[num:num+2**nb]
end = time.time()

print(f'Solucion con todo simulado: {sol/np.linalg.norm(sol)}')
print('Tiempo:',end-start)