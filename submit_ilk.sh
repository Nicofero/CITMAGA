#!/bin/bash
#
#    Modify this to your needs
#
#SBATCH -p ilk
#SBATCH -N 1
#SBATCH -o ./outputs/ilk-%j.out
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH -t 5:0:0
#SBATCH --mem-per-cpu=1G

module load qmio-tools
module load gcc qiskit

# python statevectors_test.py
# python hybrid.py
python pruebas.py