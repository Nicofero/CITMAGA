from qiskit import transpile, QuantumCircuit
from qmiotools.integrations.qiskitqmio import QmioBackend

qc = QuantumCircuit(1)
qc.x(0)
qc.measure_all()
b_qmio = QmioBackend()

qc = transpile(qc,b_qmio,optimization_level=2)

job = b_qmio.run(qc,shots=8192,repetition_period=0.01)

result = job.result().get_counts()

print(result)