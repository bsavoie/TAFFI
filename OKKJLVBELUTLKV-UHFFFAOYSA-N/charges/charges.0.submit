#!/bin/bash
#
#SBATCH --job-name charges.0
#SBATCH -o charges.0.out
#SBATCH -e charges.0.err
#SBATCH -A standby
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 4:00:00

# Prepend MPI path
#export PATH="/home/bsavoie/apps/openmpi/3.0.1/bin:$PATH"
#export LD_LIBRARY_PATH="/home/bsavoie/apps/openmpi/3.0.1/lib:$LD_LIBRARY_PATH"

# Write out some information on the job
echo Running on hosts: $SLURM_NODELIST
echo Running on $SLURM_NNODES nodes.
echo "Running on \$SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo "Copying input file to scratch..."

# USER SUPPLIED SHELL COMMANDS

cd .
mpirun -np 24 /depot/bsavoie/apps/lammps/exe/lmp_mpi_180501 -in charges.in.init >> charges.in.out &
cd /scratch/brown/seo89/linear_molecules/nitrogen_new/OKKJLVBELUTLKV-UHFFFAOYSA-N/charges

wait
