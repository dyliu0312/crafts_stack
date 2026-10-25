#!/bin/bash

#SBATCH --job-name=merge_stack_result
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=xahcnormal
#SBATCH --output=log_merge_stack_result%j.log

pwd; hostname; date

# load the environment
source ~/miniconda3/bin/activate astro

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

# --- Required Parameters---
export OUTPATH="/work/home/liudy/data/stack_result/galaxy_pair_stack/merged_stack_result_random_pca.h5"
export BASE="/work/home/liudy/data/liudy/pair_stack_result"
export FNAME_PATTERN="{}_df{}_{}_xsize{}_nfs{}_{}.h5"
export DSET_KEYS='Signal,Mask'
export AVERAGE=true

xsize='3000'
nfs='10'
seed="1 2 3 4 5 6 7 8 9 42"
name='random_crafts'
df_size='122k'
method='arpls_rmfg rmfg15 rmfg30'

export ARGS="${name},${df_size},${method},${xsize},${nfs},${seed}"
python /work/home/liudy/mycode/stack/merge_stack_result.py

echo "---------------------Merge stacked results fished--------------------------"
