#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=08:00:00

# ASk for some GPU
#$ -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=8G
#$ -pe smp 5

# Send emails when job starts and ends
#$ -m be

# Now run the job
#module add anaconda
module add cuda 
conda activate GraphNetworks
python main_bayesian.py --net_type=3conv3fc --dataset=CIFAR10
#python bayesian_approach.py


