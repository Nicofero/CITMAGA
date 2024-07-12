from hhl_functions import *
from qmiotools.integrations.qiskitqmio import QmioBackend, FakeQmio

def ampl_from_sim(qc,backend,nb,nl,shots=8192,reps=1):
    qc.remove_final_measurements()
    qc.measure_all()
    qc = transpile(qc,backend,optimization_level=2)  

    counts = {}
    for _ in range(reps):
        job = backend.run(qc,shots=shots)
        job_result = job.result()
        res=job_result.get_counts()
        counts = {k: counts.get(k, 0) + res.get(k, 0) for k in set(res) | set(counts)}

    all_outcomes = [''.join(outcome) for outcome in product('01', repeat=nb+nl+2)]

    prob = {}
    for elem in all_outcomes:
        if elem in counts:
            prob[elem]=np.sqrt(counts[elem]/shots)
        else:
            prob[elem]=0
    return prob

print("\n\n-------------PRUEBA VECTOR B-------------\n\n")

nb=2
NB = 2**nb
matrix = tridiag_matrix(2,-1,NB)

b_side = b_state(nb,'x',0.01)
st = Statevector(b_side).data.real
print(st[4:]/np.linalg.norm(st[4:]),"\n")

backend = QmioBackend()
backend2 = FakeQmio()

hhl = HHL(matrix,nb,fnc='x')

print(hhl.solution(flag=True))

vect = ampl_from_sim(hhl.qc,backend2,nb,4)

print(vect)

arr=[]
arr.append(vect['11000000'])
arr.append(vect['11000001'])
arr.append(vect['11000010'])
arr.append(vect['11000011'])
print(arr/np.linalg.norm(arr))