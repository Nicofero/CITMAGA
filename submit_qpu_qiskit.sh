#!/usr/bin/env sh

#SBATCH -p qpu
#SBATCH --mem 4G
#SBATCH -t 00:10:00

module load qmio-tools
module load gcc qiskit

python pruebas.py
