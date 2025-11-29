#!/bin/bash

#SBATCH --job-name=cut_cat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=xahcnormal
#SBATCH --output=log_sample_random_paircat%j.log

pwd; hostname; date

# load the environment
source ~/miniconda3/bin/activate astro

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

############### set parameters ##################

## query sdss catalog
export CATALOG_INPUT="/work/home/liudy/data/sdss_catalog/random_main_sample_for_crafts_pair.h5"
export N_SAMPLES="257397"

for i in {0..9}
do
    export RANDOM_SEED=$i
    echo "Processing sample $i with random seed $RANDOM_SEED"
    export CATALOG_OUTPUT="/work/home/liudy/data/sdss_catalog/random_crafts_skyarea_galaxy_pair_catalog_${i}.h5"

    python -u /work/home/liudy/mycode/stack/catalog_sample.py

    echo "Completed sample $i"
    echo "----------------------------------------"
done

echo "All samples processed successfully!"

date
