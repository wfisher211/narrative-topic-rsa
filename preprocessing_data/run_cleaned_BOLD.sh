#!/bin/sh
#SBATCH --account=psych
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=clean_bold
#SBATCH --mail-type=END
#SBATCH --mail-user=wf2315@columbia.edu
#SBATCH --output=slurm/slurm_%x_%a_%j.out
#SBATCH --array=0-5

module purge #start clean
module load singularity #load modules


# Stimulus list 
STIMS=(slumlordreach pieman black forgot reachforstars notthefallintact)
STIM=${STIMS[$SLURM_ARRAY_TASK_ID]}

# set paths
CONTAINER=/burg/psych/users/software/fajardgb_pyfmri.sif

# run the script w the singularity container
singularity exec ${CONTAINER} python -u cleaned_BOLD.py --stim "$STIM"

echo '****** DONE *******'