#!/bin/bash

#SBATCH --job-name=halofit_jk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=xahcnormal
#SBATCH --output=log_halofit_analytical_jk%j.log

pwd; hostname; date

# load the environment
source ~/miniconda3/bin/activate astro

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

############### set parameters ##################
# PCA cleaning
name='crafts_df122k'
mode='1 5 10 15 20 25 30 35 40 45'
xsize=3000
nfs=10

export PATTERN="{}_rmfg{}_xsize{}_nfs{}.h5"
export ARGS="${name},${mode},${xsize},${nfs}"

export UNIT_FACTOR="1e6" # K to microK

export BASE="/work/home/liudy/data/liudy/pair_stack_result/"

export OUTPUT="/work/home/liudy/data/stack_result/galaxy_pair_stack/halofit_jk/crafts_df122k_xsize3000_nfs10_mean_analytical_result.h5"

export WEIGHT="/work/home/liudy/data/stack_result/galaxy_pair_stack/pixcount_xsize3000_nfs10.h5"
export WEIGHT_KEY="crafts_122k/weight"

echo "Running data processing script..."
python -u /work/home/liudy/mycode/stack/halofit_analytical_jk.py
echo "------------------------fished--------------------------"

date

# airPLS fgrm method
name='crafts_df122k_arpls_rmfg'
xsize=3000
nfs=10

export PATTERN="{}_xsize{}_nfs{}.h5"
export ARGS="${name},${xsize},${nfs}"

export UNIT_FACTOR="1e6" # K to microK

export BASE="/work/home/liudy/data/liudy/pair_stack_result/"

export OUTPUT="/work/home/liudy/data/stack_result/galaxy_pair_stack/halofit_jk/crafts_df122k_xsize3000_nfs10_mean_analytical_result.h5"

export WEIGHT="/work/home/liudy/data/stack_result/galaxy_pair_stack/pixcount_xsize3000_nfs10.h5"
export WEIGHT_KEY="crafts_122k/weight"

echo "Running data processing script..."
python -u /work/home/liudy/mycode/stack/halofit_analytical_jk.py
echo "------------------------fished--------------------------"

date
