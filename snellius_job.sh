#!/bin/bash

#SBATCH -n 10
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=60G
#SBATCH --job-name=run_input_type
#SBATCH --output=slurm/output_input_type_%A.out

module load 2023
module load Anaconda3/2023.07-2
module load Java/11.0.20 
# Your job starts in the directory where you call sbatch 
cd ../../projects/0/prjs0996/vidore-benchmark
source activate ViDoRe



export CUDA_VISIBLE_DEVICES=0  # Only use GPU 0
# Add both the current directory and src directory to PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

model_type=$1
input_type=$2
dataset=$3
python scripts/test_doc_rag.py --model "$model_type" --document_input "$input_type" --dataset "$dataset" 

