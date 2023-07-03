#!/bin/bash 
#SBATCH --ntasks=1 
#SBATCH --account=def-ravh011
#SBATCH --job-name=acrolein_cnf
#SBATCH --time=3:15:00 
#SBATCH --output=out_acrolein_cnf.log 

module load python/3.9.8 
source $HOME/.virtualenvs/cnf_ofdft/bin/activate


python acrolein_densitty_fitting_KLrev.py 
