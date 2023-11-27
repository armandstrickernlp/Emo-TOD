#!/bin/bash
#SBATCH --job-name=emo
#SBATCH --output=local_logs/emo.out 
#SBATCH --error=local_logs/emo.err 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --hint=nomultithread 
#SBATCH --account=rqz@a100
#SBATCH --constraint=a100
#SBATCH --array=0-4

# args=()

# for seed in 42 43 44 45 46
# do 
#     args+=("--seed ${seed}")
# done

srun python lora_train_gen.py --lr=4e-5 --train_data_dir=data/lm_data/txt_data/emo --eval_data_json=data/lm_data/emo.json --eval_split=valid  

# ${args[${SLURM_ARRAY_TASK_ID}]}
