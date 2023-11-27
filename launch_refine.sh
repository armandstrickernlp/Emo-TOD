#!/bin/bash
#SBATCH --job-name=refine_prev
#SBATCH --output=local_logs/refine_prev.out 
#SBATCH --error=local_logs/refine_prev.err 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread 
#SBATCH --account=rqz@a100
#SBATCH --constraint=a100


srun python refine.py --gen_outputs=<path_to_emo_gen_outputs> --context=<path_to_gen_outputs>