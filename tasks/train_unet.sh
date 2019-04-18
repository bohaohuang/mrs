#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,03,04]
#SBATCH -p collinslab --gres=gpu:1
module load Python-GPU/3.6.5
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/mrs
LR=1e-4
LRSTR=1e4
EP=20
SD=/work/bh163/Models/pt_resunet_lr${LRSTR}_ep${EP}/
python train_unet.py --init-lr=${LR} --epochs=${EP} --save-dir=${SD}model.pt --log-dir=${SD} \
 --data-file=/work/bh163/mrs/inria/file_list.txt --gpu=0