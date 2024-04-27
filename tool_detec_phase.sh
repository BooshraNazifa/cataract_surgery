#!/bin/bash
#SBATCH --job-name=tool-phase-ground
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=140G
#SBATCH --time=25:00:00
#SBATCH --output=tool_phase_ground-%j.out
#SBATCH --account=def-holden

module load python/3.10
module load scipy-stack
module load cuda/12.2

nvidia-smi
source /home/booshra/final_project/myenv/bin/activate

# Set the master address and port (use the first node's hostname)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=8888

# Run the script using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /home/booshra/final_project/cataract_surgery/tool_detection_for_phase.py
nvidia-smi

deactivate