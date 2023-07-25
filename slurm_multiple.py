import os
import time
# import numpy as np

def sh_file(obs, N,l,beta_):
    f_tail = '{}_N_{}_l_{}_{}'.format(obs,N,l,beta_) # name of the file to save 

    f=open('JC_%s.sh'%(f_tail), 'w+')
    f.write('#!/bin/bash \n')
    f.write('#SBATCH --ntasks=1 \n')

    f.write('#SBATCH --account=def-ravh011\n')
    f.write('#SBATCH --job-name={} \n'.format(f_tail))
    f.write('#SBATCH --time=0:12:00 \n') # time of computation
    f.write('#SBATCH --output=out_{}.log \n'.format(f_tail))

    f.write('\n')
#     LOAD MODULES
    f.write('module load python/3.9.8 \n')
    f.write('source $HOME/jaxenv/bin/activate\n') # load your environment
    f.write('module load python/3.9.8 \n')

    for a in alpha:
        f.write('python <FILE>.py --model {} --name {} --alpha {} --thr {} \n'.format(put the variables))

    f.write('\n')

    f.write('\n')
    f.close()

    if os.path.isfile('sindy_%s.sh'%(f_tail)):
        print('Submitting sindy_%s.sh'%(f_tail))
        os.system('sbatch sindy_%s.sh '%(f_tail))

def main():
#    beta_ = 'exp_freezeR'
#    print('caca')
#    sh_file(5,0,beta_)
#    assert 0

    beta_ = 'c'
    obs = 'homo_lumo'
    n_ = [5,50,25,10,5]
    for n in n_[:1]:
        for l in range(0,25):
            sh_file(obs,n,l,beta_)
            assert 0


if __name__== "__main__":
    main()
    