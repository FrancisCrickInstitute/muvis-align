#!/usr/bin/env bash
#SBATCH --job-name=muvis_align-empiar12193
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=640G   # Memory pool for all cores (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate muvis-env
python run.py --params=resources/params_EMPIAR12193.yml