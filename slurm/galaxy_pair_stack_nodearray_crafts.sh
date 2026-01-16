#!/bin/bash

#SBATCH --job-name=pair_stack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=128:00:00
#SBATCH --partition=xahcnormal
#SBATCH --array=0-9
#SBATCH --output=log_galaxy_pair_stack_%A_%a.log

pwd; hostname; date

# load the environment
source ~/miniconda3/bin/activate astro

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

############### set parameters ##################

# PCA rm mode
modes=(1 5 10 15 20 25 30 35 40 45)

# select base on task id
mode_index=$SLURM_ARRAY_TASK_ID
mode=${modes[$mode_index]}

# Stacking parameters
export NFS=10             # Number of frequency slices
export NWORKER=64         # Number of workers for multiprocessing (should be <= --cpus-per-task)
export SSIZE=500          # Split size for pair catalog processing
export RANDOM_FLIP="True" # Randomly flip individual pair map (True/False)
export HALFWIDTH="3.0"    # Stack result map half-width
export NPIX_X="120"       # Stack result map X pixels
export NPIX_Y="120"       # Stack result map Y pixels

# Define base paths and prefixes
export INPUT_MAP_BASE="/work/home/liudy/data/liudy/prepared_mapcube/CRAFTS_PCA/"
export INPUT_MAP_KEYS="T,mask,f_bin_edge,x_bin_edge,y_bin_edge"
export INPUT_MAP_MASKED="True" # True to read 'mask' dataset in the input map file. Change to false to directly use zore masking.

export INPUT_PAIRCAT_BASE="/work/home/liudy/data/sdss_catalog/"
export INPUT_PAIRCAT_PREFIX="crafts_skyarea_galaxy_pair_catalog.h5"
export INPUT_PAIRCAT_KEYS='is_ra,pos'

export OUTPUT_STACK_BASE="/work/home/liudy/data/liudy/pair_stack_result/"
export OUTPUT_STACK_DATA_KEYS='Signal,Mask'

xsize=3000

echo "--------------------------------------------------------"
echo "Processing map data with $mode foreground modes removed."
echo "Task ID: $SLURM_ARRAY_TASK_ID, Node: $SLURMD_NODENAME"

export INPUT_MAP_PREFIX='prepared_df122k_rmfg'$mode'_xsize'$xsize'.h5'
export OUTPUT_STACK_PREFIX='crafts_df122k_rmfg'$mode'_xsize'$xsize'_nfs'$NFS'.h5'

# Run the Python
python /work/home/liudy/mycode/stack/pair_stack.py
echo "------------------------finished--------------------------"

date
