from hhl_functions import *
import matplotlib.pyplot as plt

gates={}
for i in range(4):
    size = 2**(i+1)
    vector = np.array([1]*size)
    matrix = tridiag_matrix(2,-1,size)
    hhl = HHL(matrix,vector,flag=True)
    count = hhl.qc.decompose(reps=6).count_ops()
    tot_gates=-3
    for elem in count.items():
        tot_gates += elem[1]
    gates[size]=tot_gates
    print(f'Number of gates in circuit of size {size}: {tot_gates}')

gates_est={}
for i in range(4):
    size = 2**(i+1)
    vector = np.array([1]*size)
    matrix = tridiag_matrix(2,-1,size)
    hhl = HHL(matrix,vector,flag=False)
    count = hhl.qc.decompose(reps=6).count_ops()
    tot_gates=-3
    for elem in count.items():
        tot_gates += elem[1]
    gates_est[size]=tot_gates
    print(f'Number of gates in circuit of size {size}: {tot_gates}')

keys_1 = list(gates.keys())
values_1 = list(gates.values())
keys = list(gates_est.keys())
values = list(gates_est.values())

plt.figure(figsize=(10, 7))

plt.plot(keys_1, values_1, marker='o', linestyle='-', color='blue')
plt.plot(keys, values, marker='o', linestyle='-', color='red')
plt.yscale('log')
plt.xlabel('Size')
plt.ylabel('Gates')
plt.title('Comparation in the number of needed gates')
plt.legend(['Exact Rotation','Chebyshev approximation'])
plt.grid(True)
plt.savefig("gates-png")