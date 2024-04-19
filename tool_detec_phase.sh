#!/bin/bash
#SBATCH --job-name=tool-phase-ground
#SBATCH --account=def-holden
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=140G
#SBATCH --output=tool_phase_ground-%j.out

module load python/3.10

nvidia-smi
source /home/booshra/final_project/myenv/bin/activate

python /home/booshra/final_project/cataract_surgery/tool_detection_for_phase.py
nvidia-smi
deactivate