#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4 # request a GPU
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=183100M
#SBATCH --time=45:00:00
#SBATCH --output=../output_logs/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=jnoat92@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# salloc --time=3:00:0 --account=def-dclausi --nodes 1 --tasks-per-node=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32G

echo "No. of task per node: $SLURM_NTASKS"

module purge
module load python/3.9.6
module load cuda        # in case of using numba library

#module load scipy-stack
echo "Loading module done"
source ~/torch_magic1/bin/activate
echo "Activating virtual environment done"

echo "Executing..."
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

cd /home/jnoat92/projects/def-dclausi/jnoat92/IRGS_Trans/Codes
srun python Main_Script_Executor.py --model_id $1 --train $2 --exp $3 --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS --num_workers $SLURM_CPUS_PER_TASK --nodes $SLURM_NNODES
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
