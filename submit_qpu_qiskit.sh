#!/usr/bin/env sh

#SBATCH -p qpu
#SBATCH --mem 4G
#SBATCH -t 00:10:00
#SBATCH -o ./outputs/qpu-%j.out

module load qmio-tools
module load gcc qiskit

# python hybrid.py
# python pruebas.py
python b_test.py