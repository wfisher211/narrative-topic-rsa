#!/bin/sh
#SBATCH --account=psych
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=int_msk
#SBATCH --mail-type=END
#SBATCH --mail-user=wf2315@columbia.edu
#SBATCH --output=slurm/slurm_%x_%a_%j.out
#SBATCH --array=1-1

module purge #start clean
module load singularity #load modules

# set paths
CONTAINER=/burg/psych/users/software/fajardgb_pyfmri.sif

# run the script w the singularity container
singularity exec ${CONTAINER} python -u intersect_mask.py

echo '****** DONE *******'