# Nicolas Fernandez Otero - USC
# 
# Creation of a circuit that implements HHL for Toeplitz symmetrical tridiagonal matrix. 
# 
# This notebook is based in 2 papers:
#
# - [1] : Harrow, A. W., Hassidim, A., Lloyd, S. (2009). Quantum algorithm for linear systems of equations. Phys. Rev. Lett. 103, 15 (2009), 1–15. <https://doi.org/10.1103/PhysRevLett.103.150502>
# - [2] : Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020). Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`
#
# The code is mostly based in the original code for the HHL solver in Qiskit-Algorithms

import numpy as np
from typing import Optional
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile, ClassicalRegister
from qiskit.circuit.library import PhaseEstimation, RYGate
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev  
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from numpy_matrix import NumPyMatrix
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from IPython.display import display, Latex, Math

class HHL():

    def __init__(self,matrix,vector, tolerance: float = 10e-3, flag: bool = True, meas: bool = False,fnc: str = None,c:float = None) -> None:
        """
        Builds the HHL circuit using the required args
        
        Args:
            `matrix`: The matrix that defines the linear system, i.e. A in Ax = b.
            `vector`: The right-hand side of the equation, i.e. b in Ax = b.
            `tolerance`: Tolerance of the solution bounds. This value is used to define the 3 tolerances needed for the HHL [2] equation (62).
            `flag`: Flag deciding whether the reciprocal circuit is or not exact
            `meas`: Flag deciding whether measures are made in the non x qubits
            `fnc`: String selecting the function to map the values to b. The style is: 'ax^n+...+bx+c'
            
        Returns:
            An HHL object containing the circuit in obj.qc
        
        Raises:
            ValueError: If the data is not in the right format
            ValueError: The matrix dimension is not correct
        """

        # Setting parameters
        self.matrix = matrix
        self.vector = vector
        self.tolerance = tolerance
        self.flag = flag
        self.meas = meas
        self.fnc = fnc
        self.c = c
        self.anc = False
        self.scaling = 1

        self.qc = self.build_circuit(matrix,vector)

    def __str__(self):
        string = f'nb = {self.nb}\nnl = {self.nb}'
        return string

    # Needed function to build the circuit
    def get_delta(self, lambda_min: float, lambda_max: float) -> float:
        """Calculates the scaling factor to represent exactly lambda_min on nl binary digits.

        Args:
            n_l: The number of qubits to represent the eigenvalues.
            lambda_min: the smallest eigenvalue.
            lambda_max: the largest eigenvalue.

        Returns:
            The value of the scaling factor.
        """
        formatstr = "#0" + str(self.nl + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2 ** self.nl - 1) / lambda_max)
        # floating point precision can cause problems
        if np.abs(lambda_min_tilde - 1) < 1e-7:
            lambda_min_tilde = 1
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i, char in enumerate(binstr):
            lamb_min_rep += int(char) / (2 ** (i + 1))
        return lamb_min_rep
    

    #Function to build the HHL circuit
    def build_circuit(self,matrix, vector):
        """
        Builds the HHL circuit using the required args
        
        Args:
            `matrix`: The matrix that defines the linear system, i.e. A in Ax = b.
            `vector`: The right-hand side of the equation, i.e. b in Ax = b.
            
        Returns:
            The HHL circuit
        
        Raises:
            ValueError: If the data is not in the right format
            ValueError: The matrix dimension is not correct
        """
        
        # Define the tolerances of the circuit
        epsilon_a = self.tolerance/6
        epsilon_r = self.tolerance/3
        epsilon_s = self.tolerance/3
        
        # We need an np.array to write the values to the register
        if isinstance(vector,(list,np.ndarray,int)):
            if isinstance(vector,list):
                vector = np.array(vector)
            
            # We define the number of needed qubits and insert the vector to the register
            
            if isinstance(vector,int):
                self.nb = vector
            else:
                self.nb = int(np.log2(len(vector)))
            
            if self.fnc is None:            
                vector_circuit = QuantumCircuit(self.nb)
                vector_circuit.initialize(vector / np.linalg.norm(vector), list(range(self.nb)), None)
            
            else:
                if self.c is None:
                    # Could be calculated in O(N) time, so we input an approximation of Cb to calculate in O(1)
                    Cb = 0.9
                    self.c = epsilon_s/(8*Cb)
                    vector_circuit = b_state(self.nb,self.fnc,self.c)
                else:
                    vector_circuit = b_state(self.nb,self.fnc,self.c)
                self.anc = True
        else:
            raise ValueError(f"Invalid type for vector: {type(vector)}.")
        
        # Define flag, if 1, correct solution, if 0, incorrect
        self.nf = 1
        
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
            
            # if matrix.shape[0] != 2 ** vector_circuit.num_qubits:
            #     raise ValueError(
            #         "Input vector dimension does not match input "
            #         "matrix dimension! Vector dimension: "
            #         + str(vector_circuit.num_qubits)
            #         + ". Matrix dimension: "
            #         + str(matrix.shape[0])
            #     )
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
        self.nl = max(self.nb+1,int(np.log2(kappa))+1)
        
        # Define eigenvalues
        if hasattr(matrix_circuit, "eigs_bounds"):
            lambda_min, lambda_max = matrix_circuit.eigs_bounds()
            # Constant so that the minimum eigenvalue is represented exactly, since it contributes
            # the most to the solution of the system
            delta = self.get_delta(lambda_min, lambda_max)

            # Update evolution time
            matrix_circuit.evolution_time = 2 * np.pi * delta / lambda_min
            # Update the scaling of the solution
            self.scaling = lambda_min
        else:
            delta = 1 / (2 ** self.nl)
            print("The solution will be calculated up to a scaling factor.")
        
            
        # Define the reciprocal circuit
        
        # # Using an exact reciprocal circuit
        if self.flag:
            reciprocal_circuit = ExactReciprocal(self.nl, delta)
            # Update number of ancilla qubits
            self.na = matrix_circuit.num_ancillas
        
        # Using Chebyshev interpolation to approximate arcsin(C/x) to a degree of degree
        else: 
            # Calculate breakpoints for the reciprocal approximation
            num_values = 2 ** self.nl
            constant = delta
            
            # No tengo para nada claro esto, no encuentro que hay que hacer con la a para pasarla a entero
            a = int(2**(2*self.nl/3))  # pylint: disable=invalid-name

            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(self.nb,int(np.log(1+(16.23* np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2)* kappa* (2 * kappa - epsilon_r))/ epsilon_r)),)
            # As [2]
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))

            # Calculate breakpoints and polynomials
            breakpoints = []
            for i in range(0, num_intervals):
                # Add the breakpoint to the list
                breakpoints.append(a * (5 ** i))

                # Define the right breakpoint of the interval
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)
            # Once we have the intervals, and everything is defined, we can make an approximation by a polynomial function of degree
            reciprocal_circuit = PiecewiseChebyshev(
                lambda x: np.arcsin(constant / x), degree, breakpoints, self.nl, name="c_Rot"
            )
            # Number of ancilla qubits
            self.na = max(matrix_circuit.num_ancillas, reciprocal_circuit.num_ancillas)
            
        # Construction of the circuit
            
        # Initialise the quantum registers
        qb = QuantumRegister(self.nb,name="b")  # right hand side and solution
        ql = QuantumRegister(self.nl,name="0")  # eigenvalue evaluation qubits
        if self.na > 0:
            qa = AncillaRegister(self.na,name="anc")  # ancilla qubits
        if self.anc:
            qab = AncillaRegister(1,name='a_b') # ancilla qubit for the approximation of b
        qf = QuantumRegister(self.nf,name="flag")  # flag qubits

        if self.na > 0:
            if self.anc:
                qc = QuantumCircuit(qb, ql, qa, qab, qf)
            else:
                qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            if self.anc:
                qc = QuantumCircuit(qb, ql, qab, qf)
            else:
                qc = QuantumCircuit(qb, ql, qf)

        # State preparation
        if self.anc:
            qc.append(vector_circuit, qb[:] + qab[:])
        else:
            qc.append(vector_circuit, qb[:])
        qc.barrier(label="\pi_1")
        # QPE
        phase_estimation = PhaseEstimation(self.nl, matrix_circuit)
        if self.na > 0:
            qc.append(phase_estimation, ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation, ql[:] + qb[:])
        qc.barrier(label="\pi_2")
        # Conditioned rotation
        if self.flag:
            qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(
                reciprocal_circuit.to_instruction(),
                ql[::-1] + [qf[0]] + qa[: reciprocal_circuit.num_ancillas],
            )
        
        qc.barrier(label="\pi_3")
        # QPE inverse
        if self.na > 0:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:])
        
        if self.meas:
            qc.measure_all()
        return qc
    
    # Calculates the norm of the solution
    def calculate_norm(self) -> float:
        """Calculates the value of the euclidean norm of the solution.
        Returns:
            The value of the euclidean norm of the solution.
        """        
        # Get the state of the circuit
        statevector = Statevector(self.qc)
        st=np.array(statevector).real
        num = int(len(st)/2)
        
        prob = sum(st[num:]**2)
        norm_b = np.linalg.norm(self.vector)
        norm_x = np.sqrt(prob)*norm_b/self.scaling

        return norm_x

    def solution(self, flag: bool = False,norm: bool = True) -> np.ndarray:
        """Returns the solution for the given HHL Quantum Circuit using Statevectors. You can obtain the solution from the circuit or the real solution depending on `norm` flag
        
        ## Args:
            `qc`: The HHL circuit to be solved
            `flag`: The flah indicating if the circuit is 
            `norm`: Indicates if you want the solution to the normalized probem (True) or the solution to the full problem (False). True by default
        ## Returns:
            A numpy array containing the solution
        """
        self.qc.remove_final_measurements()
        st = Statevector(self.qc).data.real
        length = self.na + self.nb + self.nl + self.nf
        
        if flag:
            num = int(len(st)/2) + 2**(length-2)
        else:
            num = int(len(st)/2)

        prob = sum(st[num:]**2)
        st = st[num:num+2**self.nb]
        sol = st/np.linalg.norm(st)
        if not norm:  
            norm_b = np.linalg.norm(self.vector)
            norm_x = np.sqrt(prob)*norm_b/self.scaling
            sol = norm_x*sol
        return sol

    def prob_from_counts_hhl(self,counts) -> np.ndarray:
        """ Calculates the expected probability of the solution |x> without normalization
        
        ## Args:
            `counts`: Counts as a dictionary {'xxxxx': number}, obtained from a simulation or run in a real QPU
        ## Returns:
            The non normalized probabilities of the solution. To get the real solution, you should get its square root, normalize it and multiply it by the norm of the solution.
        """ 
        all_outcomes = [''.join(outcome) for outcome in product('01', repeat=self.nb)]

        # Initialize the dictionary with each binary string as a key and a value of 0
        prob_amplitudes = {outcome: 0 for outcome in all_outcomes}

        shots = sum(counts.values())

        for outcome, count in counts.items():
            first_qubit_state = outcome[-self.nb:]  # Get the state of the first qubit
            prob_amplitudes[first_qubit_state] += count / shots

        ampl = np.array(list(prob_amplitudes.values()))
        return ampl
    
    # Transpile simulate and get counts from a circuit
    def get_counts(self,backend,shots:int = 8192):
        """Transpile, simulate and get counts from a circuit
        
        ## Args:
            `qc`: Quantum circuit to simulate
        
        ## Returns:
            Dictionary containing a number of counts for each key
        """
        # Remove and add measurements, to not have multiple measurements
        self.qc.remove_final_measurements()
        self.qc.measure_all()

        # Define simulator
        qc_meas = transpile(self.qc,backend,optimization_level=2)

        job = backend.run(qc_meas,shots=shots)
        job_result = job.result()

        counts=job_result.get_counts()

        return counts
    
    # Gets the norm of the original vector from counts
    def norm_from_counts(self,counts):
        filtered_counts = {k: v for k, v in counts.items() if k[0] == '1'}
        total_filtered_counts = sum(filtered_counts.values())
        norm_b = np.linalg.norm(self.vector)
        shots = sum(counts.values())
        P1 = total_filtered_counts/shots
        norm_x = np.sqrt(P1)*norm_b/(self.scaling)
        return norm_x

#################################################################################################################

def parse_func(poly_str):
    # Remove spaces
    poly_str = poly_str.replace(' ', '')
    
    # Standardize the polynomial string to handle positive terms properly
    poly_str = poly_str.replace('-', '+-')
    if poly_str[0] == '+':
        poly_str = poly_str[1:]
    
    # Split the string into terms
    terms = poly_str.split('+')
    
    coefficients = {}
    max_degree = 0
    
    for term in terms:
        if 'x' in term:
            if '^' in term:
                coef, exp = term.split('x^')
                exp = int(exp)
            else:
                coef, exp = term.split('x')
                exp = 1
            
            if coef in ('', '+'):
                coef = 1
            elif coef == '-':
                coef = -1
            else:
                coef = int(coef)
        else:
            coef = int(term)
            exp = 0
        
        coefficients[exp] = coef
        if exp > max_degree:
            max_degree = exp
    
    # Fill missing degrees with 0
    all_coefficients = [coefficients.get(i, 0) for i in range(max_degree + 1)]
    return all_coefficients,max_degree

def value_func(coefs,nb):
    
    if not isinstance(coefs,np.ndarray):
        coefs = np.array(coefs)
    
    size = 2**nb
    
    pol=[]
    for i in range(size):
        aux = 0
        for j in range(1,len(coefs)):
            aux +=coefs[j]*i**j
        pol.append(aux)     
    
    return pol

def loc_ancilla(qc: QuantumCircuit):
    i=0
    pos=0
    while True:        
        if isinstance(qc.qregs[i],AncillaRegister):
            break
        else:
            pos+= qc.qregs[i].size
            i+=1
            
    return pos
    
def ccry(qc:QuantumCircuit,theta: float,control: list,target):
    ry = RYGate(theta).control(len(control))
    if isinstance(control,list):
        if isinstance(target,list):
            qc.append(ry,control+target)
        elif isinstance(target,AncillaRegister):
            qc.append(ry,control+[loc_ancilla(qc)])
            
    return qc

def b_state(nb: int,function: str,c: float = 10e-7) -> QuantumCircuit:
    """Defines the b state from an approximation polynomic function
    
    Args:
        `nb`: The number of qubits needed to represent the vector
        `function`: A string representing the function. The style must be: 'ax^n+bx^n-1+...+z'. Where a,b,...,z are the amplitudes.
        
    """
    qr = QuantumRegister(nb,name="b")
    qa = AncillaRegister(1,name='a')
    qc = QuantumCircuit(qr,qa)
    
    qc.h(qr[:])
    
    # Processing of `function`
    ampl,D = parse_func(function)
    
    # Value of `function`
    pol = value_func(ampl,nb)
    
    if ampl[0]!=0:
        qc.ry(D*ampl[0]*c,qa)
    
    size = 2**nb
    
    for i in range(1,size):        
        if (np.floor(np.log2(i))==np.ceil(np.log2(i))):
            if pol[i]!=0:
                qc.cry(D*pol[i]*c,qr[int(np.log2(i))],qa)
        else:
            bin_aux = bin(i)[2:]
            index = [len(bin_aux) - 1 - j for j, digit in enumerate(bin_aux) if digit == '1' ]
            # I dont remember why this 2**i, but it made sense in the moment i coded it
            aux = [pol[2**i] for i in index]
            elem = pol[i]-np.sum(aux)
            if elem != 0:
                # qc.mcry(D*elem*c,index,qa) 
                qc = ccry(qc,D*elem*c,index,qa)    
    return qc

def b_from_func(function,size):
    coef,dg = parse_func(function)
    b = []
    for i in range(size):
        aux = 0
        for j in range(dg+1):
            aux+=coef[j]*((i/(size-1))**j)
        b.append(aux)
    return b

# Function to create np.arrays as tridiagonal matrix
def tridiag_matrix(diag,up,n) -> np.ndarray:
    """Returns a tridiagonal symmetrical matrix
    
    Args:
        `diag`: Value of the diagonal
        `up`: Value of the subdiagonals
        `n`: Size of the matrix
    
    Returns:
        The matrix
    """
    if n<2: 
        raise ValueError("The dimension of the matrix must be greater than 2")
    
    rows = []
    
    for i in range(n):
        row = np.zeros(n)
        if i>0:
            row[i-1]=up
        row[i]=diag
        if i<n-1:
            row[i+1]=up
            
        rows.append(row)
        
    matrix = np.array(rows)
    return matrix

def run_circuit(qc,shots,sampler):
    sampler = SamplerV2()
    job = sampler.run([qc],shots=shots)
    job_result = job.result()
    res=job_result[0].data.meas.get_counts()
    return res

def prob_from_sim(qc:QuantumCircuit,shots=8192):
    """Returns the probability of each possible value of the circuit qc
    
    Args:
        `qc`: Quantum circuit to be simulated
        `shots`: Number of shots
    
    """
    sim = AerSimulator()
    qc.remove_final_measurements()
    qc.measure_all()
    qc = transpile(qc,sim)
    sampler = SamplerV2()
    
    nb = qc.qregs[0].size
    nl = qc.qregs[1].size

    job = sampler.run([qc],shots=shots)
    job_result = job.result()
    counts=job_result[0].data.meas.get_counts()

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

def fourier_error_analysis(x:np.ndarray,y:np.ndarray,tol: float = 1e-6,n_peaks:int = None) -> None:
    """Computes the Fourier Analysis of a given data

    ## Args:
        `x`: Points where data is taken
        `y`: The values of the data
        `tol`: Tolerance of the peak detection. By default 1e-6
        `n_peaks`: Number of peaks from the Fourier Transform to be computed in the estimation. If no value is given, all peaks will be used
    ## Returns:
        The parameters of the fitting
    """
    ampl = np.fft.fft(y)

    freqs = np.fft.fftfreq(x.size, d=(x[1] - x[0]) )

    positive_freqs = freqs[:len(freqs)//2]
    positive_fft_result = np.abs(ampl)[:len(freqs)//2]

    peaks,_= find_peaks(positive_fft_result,threshold=tol)

    print(f'Picos en: {peaks}')

    power = np.abs(ampl)

    #print(ampl)

    if n_peaks is None:
        main_freqs_indices = peaks[:]
    else:
        main_freqs_indices = peaks[:n_peaks]
    main_freqs = freqs[main_freqs_indices]
    main_powers = power[main_freqs_indices]

    def model_func(t, *params):
        n = len(params) // 3
        result = np.zeros_like(t)
        for i in range(n):
            A = params[3*i]
            f = params[3*i + 1]
            phi = params[3*i + 2]
            result += A * np.cos(2 * np.pi * f * t + phi)
        return result

    initial_guess = []
    for freq, power in zip(main_freqs, main_powers):
        initial_guess.extend([power, freq, 0.0])

    try:
        popt, _ = curve_fit(model_func, x, y, p0=initial_guess)
    except Exception:
        print("\033[91m{}\033[00m \n".format('---------------------------------------------------------------------------'))
        print("\033[91m{}\033[00m".format('The number of peaks must be lower or the function wont converge'))
        raise
    
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)

    plt.plot(x, y,label='Original Signal')
    plt.title('Sampled Function')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot the magnitude of the FFT result
    plt.subplot(2, 1, 2)
    plt.plot(positive_freqs, np.abs(positive_fft_result))  # Only plot the positive frequencies
    plt.title('Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True,'both')

    # Construct the fitted signal from the optimized parameters
    y_fitted = model_func(x, *popt)

    # Step 5: Compare the approximation to the original vector
    plt.subplot(2, 1, 1)
    plt.plot(x, y_fitted, label='Fitted Signal', color='red')
    plt.title('Fitted Signal')
    plt.legend()

    plt.tight_layout()

    plt.show()

    print("Fitted Parameters (Amplitude, Frequency, Phase):")
    for i in range(len(popt)//3):
        print(f"Cosine {i+1}: Amplitude = {popt[3*i]:.2f}, Frequency = {popt[3*i + 1]:.2f} Hz, Phase = {popt[3*i + 2]:.2f} radians")

    print('Function:')
    latex_str = r"f(t)="
    for i in range(len(popt)//3):
        amplitude = f"{popt[3*i]:+0.2f}"
        frequency = f"{popt[3*i + 1]:.2f}"
        phase = f"{popt[3*i + 2]:+0.2f}"
        latex_str += rf" {amplitude}\cos(2\pi {frequency}t {phase})"
    display(Latex(latex_str))
    display(Math(latex_str))

    return popt

#Needed for Qmio
def get_delta(nl:int, lambda_min: float, lambda_max: float) -> float:
    """Calculates the scaling factor to represent exactly lambda_min on nl binary digits.

    Args:
        n_l: The number of qubits to represent the eigenvalues.
        lambda_min: the smallest eigenvalue.
        lambda_max: the largest eigenvalue.

    Returns:
        The value of the scaling factor.
    """
    formatstr = "#0" + str(nl + 2) + "b"
    lambda_min_tilde = np.abs(lambda_min * (2 ** nl - 1) / lambda_max)
    # floating point precision can cause problems
    if np.abs(lambda_min_tilde - 1) < 1e-7:
        lambda_min_tilde = 1
    binstr = format(int(lambda_min_tilde), formatstr)[2::]
    lamb_min_rep = 0
    for i, char in enumerate(binstr):
        lamb_min_rep += int(char) / (2 ** (i + 1))
        return lamb_min_rep

def prepare_circ(qc:QuantumCircuit):
    qc = qc.decompose(reps=8)
    qc.data = [instruction for instruction in qc.data if instruction.operation.name!='reset']
    return qc