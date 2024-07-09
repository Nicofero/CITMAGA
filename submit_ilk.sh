#!/bin/bash
#
#    Modify this to your needs
#
#SBATCH -p ilk
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH -t 5:0:0
#SBATCH --mem-per-cpu=4G

module load qmio-tools
module load gcc qiskit

python pruebas.py