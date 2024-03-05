#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=127510M
#SBATCH --time=00:59:00
#SBATCH --output=../output_logs/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=jnoat92@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# salloc --time=3:00:0 --account=def-dclausi --nodes 1 --tasks-per-node=1 --gpus-per-node=1 --cpus-per-task=28 --mem=127518M

echo "No. of task per node: $SLURM_NTASKS"

module purge
module load python/3.9.6
module load cuda/11.7        # in case of using numba library

#module load scipy-stack
echo "Loading module done"
source ~/torch_magic1/bin/activate
echo "Activating virtual environment done"

echo "Executing..."

cd /home/jnoat92/projects/def-dclausi/jnoat92/IRGS_Trans/Codes
srun python Main_Script_Executor.py --model_id $1 --train $2 --exp $3 --num_workers $SLURM_CPUS_PER_TASK
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
