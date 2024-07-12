#!/bin/bash
#SBATCH -p a64
#SBATCH -N 1                      # Numero de nodos
#SBATCH --tasks-per-node=1        # Número de tareas por nodo
#SBATCH -o ./outputs/a64-%j.out
#SBATCH -c 48                     # Número de hilos por tarea
#SBATCH -t 0:10:0                 # Time limit
#SBATCH --mem-per-cpu=1G        # Memoria por cpu

source /etc/profile.d/lmod.sh

export OMP_NUM_THREADS=48
export QULACS_NUM_THREADS=48

module load qulacs-hpcx

scontrol show hostnames
echo $SLURM_JOB_NODELIST
nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n" "${nodelist[@]}" > nodefile

# Casos grandes se benefician de
# export OMP_PROC_BIND=TRUE
# numactl -N 0-3 <antes de la llamada de python>

mpirun -npernode ${SLURM_NTASKS_PER_NODE} -hostfile nodefile python qulacs_test.py
rm nodefile