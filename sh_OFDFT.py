import os

def sh_file(mol_name,kin,v_pot,h_pot,x_pot,c_pot,batch_size,epochs,lr,nn,bool_params,sched_type,optimizer,prior_distribution):

    nameTail = f'{mol_name}-{prior_distribution}-{nn}l-Ce-{c_pot}-I-{epochs}-O-{optimizer}-lr-{lr}-bs{batch_size}'

    f = open(f"alex_{nameTail}.sh", 'w+')
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --ntasks=1\n')

    f.write('#SBATCH --account=def-ravh011\n')
    #f.write(f'#SBATCH --account=rrg-ravh011\n')
    f.write(f'#SBATCH --job-name={nameTail}\n')
    f.write('#SBATCH --time=72:10:00\n') # time of computation
    f.write(f'#SBATCH --output=out_{nameTail}.log\n')
    f.write(f'#SBATCH --mem-per-cpu=90G\n')
    f.write(f'#SBATCH --gpus-per-node=v100l:1\n')
    f.write(f'#SBATCH --cpus-per-task=2\n\n') 

    f.write('module load gcc/9.3.0 cuda/11.8.0 cudnn/8.6 python/3.9 \n')
    f.write('export LD_LIBRARY_PATH=$EBROOTCUDA/lib:$EBROOTCUDNN/lib \n')
    f.write('export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME \n')
    f.write('ORIGINAL_DIR=$(pwd) \n')
    f.write('cd /home/al3x/ && mkdir alex_envs && cd alex_envs && virtualenv --no-download ENV \n')
    f.write('cd $ORIGINAL_DIR \n')
    f.write('source /home/al3x/alex_envs/ENV/bin/activate \n') # load your environment
    f.write('module load scipy-stack \n')
    f.write('pip install -r requirements.txt \n') 
    #############################################################################i
    ## WRITE SCRIPT EXECUTION

    f.write(f'python OFDFT_NF.py --mol_name {mol_name} --kin {kin} --nuc {v_pot} --hart {h_pot}  --x {x_pot} --c {c_pot} --bs {batch_size} --epochs {epochs} --lr {lr} --nn {nn}  --params {bool_params} --sched {sched_type} --opt {optimizer} --dist {prior_distribution}\n') ## ADD VARIABLES HERE
   
    f.write('\n\n')
    f.close()




    #############################################################################
    ## SUBMIT JOB TO COMPUTE CANADA

    if os.path.isfile(f"alex_{nameTail}.sh"):
        print(f"Submitting alex_{nameTail}.sh")
        os.system(f"sbatch alex_{nameTail}.sh")


if __name__== "__main__":
    
    mol_name = ['C27H46O']  
    kin = 'tf-w' #Put arguments here
    v_pot = 'nuclei_potential' #Put arguments here
    h_pot = 'hartree'#Put arguments here
    x_pot = 'dirac_b88_x_e'
    c_pot = ['pw92_c_e'] 
    batch_size = [2048]
    epochs = 20000
    lr = '3E-4'
    nn = [384]
    bool_params = False
    sched_type = 'mix'
    optimizer = ['adam']
    prior_dist = ['pro_mol']

    for _mol_name in mol_name:
        for _c_pot in c_pot:
            for _nn in nn:
                for _opt in optimizer:
                    for _prior_dist in prior_dist:
                        for _bs in batch_size:
                            sh_file(_mol_name,kin,v_pot,h_pot,x_pot,_c_pot,_bs,epochs,lr,_nn,bool_params,sched_type,_opt,_prior_dist)