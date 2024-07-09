#!/usr/bin/env sh

#SBATCH -p qpu
#SBATCH --mem 4G
#SBATCH -t 00:10:00

module load gcc qiskit
module load qmio-run

python qasm_test.py