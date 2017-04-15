#!/bin/bash
#PBS -A mta-201-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=1
#PBS -o /home/siddhu95/dcgan_tf/output.log
#PBS -l feature=k80

#source /admin/bin/migrate_softwares.sh $LSCRATCH
#virtualenv $LSCRATCH/myenv
#source $LSCRATCH/myenv/bin/activate

#nvidia-smi
module load compilers/gcc/4.8.5 cuda/7.5.18 libs/cuDNN/5
module load apps/python/2.7.10
#module load libs/torch
module load libs/theano
#source ~/ENV3/bin/activate
export PATH="/home/siddhu95/anaconda2/bin:$PATH"

echo $PYTHONPATH
echo "Starting at `date`" 
cd /home/siddhu95/dcgan_tf/
THEANO_FLAGS=device=gpu python keras_model.py > outputx3_inception.txt
echo "Finished at `date`"
