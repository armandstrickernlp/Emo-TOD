#!/bin/bash
#SBATCH --job-name=gpt2_emo
#SBATCH --output=local_logs/gpt2_emo.out 
#SBATCH --error=local_logs/gpt2_emo.err 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread 
#SBATCH --account=rqz@v100
#SBATCH --constraint=v100

 # --array=0-4


# args=()

# for seed in 42 43 44 45 46
# do 
#     args+=("--seed ${seed}")
# done

srun python gpt2_train_gen.py --lr=8e-5 --seed=42 --train_data_dir=data/lm_data/txt_data/emo --eval_data_json=data/lm_data/emo.json --eval_split=valid 

# ${args[${SLURM_ARRAY_TASK_ID}]}
