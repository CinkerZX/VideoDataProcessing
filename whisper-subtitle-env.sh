#!/bin/bash
#SBATCH --job-name=subtitle_generation     # Job name
#SBATCH --output=subtitle_%j.out           # Standard output file
#SBATCH --error=subtitle_%j.err            # Standard error file
#SBATCH --time=00:20:00                    # Time limit (HH:MM:SS)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Total number of tasks
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --mem=8GB                          # Memory per node
#SBATCH --partition=gpu                    # Specify GPU partition (if applicable)
#SBATCH --gpus-per-node=1                  # Request 1 GPU
#SBATCH --mail-user=sindy_zhou126@126.com  # Your email address
#SBATCH --mail-type=END                    # Email notification for job completion

# Load necessary modules (adjust these lines according to your cluster's specifications)
module load CUDA/12.1.1  # Load the appropriate CUDA version
module load cuDNN/8.9.2.26-CUDA-12.1.1  # Load the corresponding cuDNN version
module load Miniconda3/22.11.1-1  # Load Miniconda

# Create a new conda environment named whisper-subtitle-env if it doesn't exist
if ! conda info --envs | grep -q 'whisper-subtitle-env'; then
    conda create -n whisper-subtitle-env python=3.9 -y
fi

# conda install conda=25.1.1

# Activate the new conda environment
source activate whisper-subtitle-env

# Install compatible versions of PyTorch for CUDA support
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html


# Install Whisper and other necessary libraries
pip install --upgrade git+https://github.com/openai/whisper.git
pip install ffmpeg-python  # Install ffmpeg wrapper
pip install tqdm  # For progress bar
pip uninstall -y numpy
pip install numpy==1.26.4
# Verify NumPy installation
python -c "import numpy; print('NumPy version:', numpy.__version__)"

# Install ffmpeg using conda
conda install -c conda-forge ffmpeg -y

# Verify ffmpeg installation
echo "ffmpeg version:"
ffmpeg -version

# Check ffmpeg path
echo "ffmpeg path:"
which 

# Notify the user
echo "Environment 'whisper-subtitle-env' is set up for GPU and ready to use."


# Run your Python script
# python test_whisper_model.py
python SubtitleGeneratorGPU.py